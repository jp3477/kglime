# Python imports
from collections import defaultdict
from pathlib import Path
import json
import argparse
from urllib.request import urlretrieve

# Third-party imports
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
from tqdm import tqdm

all_relationships_query = """
    WITH RECURSIVE drug_structures_omop AS (
        SELECT
            s.id struct_id, s.name struct_name, i.identifier rxnorm_code,
            ingredient_drug_c.concept_id ingredient_concept_id,
            ingredient_drug_c.concept_name ingredient_concept_name,
            ingredient_drug_c.domain_id ingredient_concept_domain
        FROM structures s
        JOIN identifier i
            ON i.struct_id = s.id
                AND i.id_type = 'RXNORM'
        JOIN concept drug_c
            ON drug_c.concept_code = i.identifier
                AND drug_c.vocabulary_id IN ('RxNorm', 'RxNorm Extension')
        JOIN concept_relationship cr
            ON cr.concept_id_1 = drug_c.concept_id
                AND cr.relationship_id IN ('Maps to', 'Brand name of', 'RxNorm has ing')
        JOIN concept ingredient_drug_c
            ON ingredient_drug_c.concept_id = cr.concept_id_2
                AND ingredient_drug_c.concept_class_id = 'Ingredient'
    ),
    drug_indications AS (
    SELECT
            ingredient_concept_id concept_id_1, ingredient_concept_name concept_name_1, ingredient_concept_domain concept_domain_1,
            'has_indication' rel_name,
            indication_c.concept_id concept_id_2, indication_c.concept_name concept_name_2, indication_c.domain_id concept_domain_2
        FROM drug_structures_omop s
        JOIN omop_relationship o
            ON s.struct_id = o.struct_id
                AND o.relationship_name = 'indication'
        JOIN concept indication_c
            ON CAST(o.snomed_conceptid AS text) = indication_c.concept_code
                AND indication_c.vocabulary_id = 'SNOMED'

    ),
    drug_adverse_effects AS (
        SELECT
            ingredient_concept_id concept_id_1, ingredient_concept_name concept_name_1, ingredient_concept_domain concept_domain_1,
            'has_ae' rel_name,
            standard_ae_c.concept_id concept_id_2, standard_ae_c.concept_name concept_name_2, standard_ae_c.domain_id concept_domain_2
        FROM drug_structures_omop s
        JOIN adverse_reactions ar
            ON ar.drug_concept_ids = CAST(s.ingredient_concept_id AS TEXT)
        JOIN concept ae_c
            ON ae_c.concept_id = ar.omop_concept_id
        JOIN concept_relationship cr
            ON cr.concept_id_1 = ae_c.concept_id
                AND cr.relationship_id = 'Maps to'
        LEFT JOIN concept standard_ae_c
            ON standard_ae_c.concept_id = cr.concept_id_2
    ),
    drug_categories AS (
        SELECT DISTINCT
            ingredient_concept_id concept_id_1, ingredient_concept_name concept_name_1, ingredient_concept_domain concept_domain_1,
            'has_atc' rel_name,
            atc_c.concept_id concept_id_2, atc_c.concept_name concept_name_2, atc_c.domain_id concept_domain_2
        FROM drug_structures_omop s
        JOIN struct2atc sa
            ON s.struct_id = sa.struct_id
        JOIN atc a
            ON a.code = sa.atc_code
        JOIN concept atc_c
            ON a.l4_code = atc_c.concept_code
                AND atc_c.vocabulary_id = 'ATC'
        WHERE chemical_substance_count = 1
    ),
    abnormal_measurements AS (
        SELECT
            loinc_c.concept_id concept_id_1, loinc_c.concept_name concept_name_1, loinc_c.domain_id concept_domain_1,
            CASE
                WHEN l2h.loinc_scale = 'Qn' AND l2h.outcome = 'H' THEN 'associated_with_high'
                WHEN l2h.loinc_scale = 'Qn' AND l2h.outcome = 'L' THEN 'associated_with_low'
                WHEN l2h.loinc_scale = 'Ord' THEN 'associated_with_high'
                WHEN l2h.loinc_scale = 'Nom' THEN 'associated_with_high'
            END rel_name,
            snomed_c.concept_id concept_id_2, snomed_c.concept_name concept_name_2, snomed_c.domain_id concept_domain_2
        FROM loinc_to_hpo l2h
        JOIN concept loinc_c
            ON loinc_c.concept_code = l2h.loinc_id
        JOIN concept snomed_c
            ON snomed_c.concept_code = l2h.snomed_code
        WHERE (l2h.loinc_scale = 'Qn' AND l2h.outcome IN ('H', 'L'))
            OR (l2h.loinc_scale = 'Ord' AND l2h.outcome IN ('POS' ,'H'))
            OR (l2h.loinc_scale = 'Nom' AND l2h.outcome IN ('POS'))
    ),
    all_one_level_relationships AS (
        SELECT * FROM drug_indications
        UNION DISTINCT
        SELECT * FROM drug_adverse_effects
        UNION DISTINCT
        SELECT * FROM drug_categories
        UNION DISTINCT
        SELECT * FROM abnormal_measurements
    ),
    concept_parents(child_concept_id, parent_concept_id) AS (
        SELECT
            cr.concept_id_1 child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept par
        JOIN concept_relationship cr
            ON cr.concept_id_2 = par.concept_id
                AND cr.relationship_id = 'Is a'
        JOIN all_one_level_relationships chd
            ON (chd.concept_id_2 = cr.concept_id_1 AND chd.concept_domain_2 IN ('Condition', 'Drug', 'Measurement'))
        UNION ALL
        SELECT
            cr.concept_id_1 child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept par
        JOIN concept_relationship cr
            ON cr.concept_id_2 = par.concept_id
                AND cr.relationship_id = 'Is a'
        JOIN all_one_level_relationships chd
            ON (chd.concept_id_1 = cr.concept_id_1 AND chd.concept_domain_1 IN ('Condition', 'Drug', 'Measurement'))
        UNION ALL
        SELECT chd.parent_concept_id child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept_parents chd
        JOIN concept_relationship cr
            ON cr.concept_id_1 = chd.parent_concept_id
                AND cr.relationship_id = 'Is a'
    ),
    concept_children(child_concept_id, parent_concept_id) AS (
        SELECT
            cr.concept_id_1 child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept chd
        JOIN concept_relationship cr
            ON cr.concept_id_1 = chd.concept_id
                AND cr.relationship_id = 'Is a'
        JOIN all_one_level_relationships par
            ON (par.concept_id_2 = cr.concept_id_2 AND par.concept_domain_2 IN ('Condition', 'Drug', 'Measurement'))
        UNION ALL
        SELECT
            cr.concept_id_1 child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept chd
        JOIN concept_relationship cr
            ON cr.concept_id_1 = chd.concept_id
                AND cr.relationship_id = 'Is a'
        JOIN all_one_level_relationships par
            ON (par.concept_id_1 = cr.concept_id_2 AND par.concept_domain_1 IN ('Condition', 'Drug', 'Measurement'))
        UNION ALL
        SELECT cr.concept_id_1 child_concept_id, cr.concept_id_2 parent_concept_id
        FROM concept_children par
        JOIN concept_relationship cr
            ON cr.concept_id_2 = par.parent_concept_id
                AND cr.relationship_id = 'Is a'
    )
    SELECT
        *
    FROM all_one_level_relationships ac
    UNION DISTINCT
    SELECT
        chd.concept_id concept_id_1, chd.concept_name concept_name_1, chd.domain_id concept_domain_1,
        'is_a' rel_name,
        par.concept_id concept_id_2, par.concept_name concept_name_2, par.domain_id concept_domain_2
    FROM concept_parents cp
    JOIN concept chd
        ON chd.concept_id = cp.child_concept_id
    JOIN concept par
        ON par.concept_id = cp.parent_concept_id


"""


def download_drug_central_pg_dump():
    DUMP_LINK = "https://unmtid-shinyapps.net/download/drugcentral.dump.08222022.sql.gz"
    urlretrieve(DUMP_LINK, 'drug_central_pg_dump.gz')


def build_nx_graph(rel_df):
    colors = defaultdict(lambda: 'gray')
    colors.update({'Condition': 'blue', 'Drug': 'red'})
    # G = nx.DiGraph()
    G = nx.MultiDiGraph()

    rel_type_map = {}
    rel_type_count = 0

    for i, rel in tqdm(rel_df.iterrows()):
        G.add_node(rel['concept_id_1'],
                   concept_id=rel['concept_id_1'],
                   concept_name=rel['concept_name_1'],
                   title=rel['concept_name_1'],
                   domain_id=rel['concept_domain_1'])

        G.add_node(rel['concept_id_2'],
                   concept_id=rel['concept_id_2'],
                   concept_name=rel['concept_name_2'],
                   title=rel['concept_name_2'],
                   domain_id=rel['concept_domain_2'])

        rel_name = rel['rel_name']
        inv_rel_name = f"inv_{rel_name}"

        if rel_name not in rel_type_map:
            rel_type_map[rel_name] = rel_type_count
            rel_type_count += 1

            rel_type_map[inv_rel_name] = rel_type_count
            rel_type_count += 1

        j = i * 2
        G.add_edge(rel['concept_id_1'],
                   rel['concept_id_2'],
                   id=j,
                   title=rel_name,
                   rel_name=rel_name,
                   rel_type=rel_type_map[rel_name])

        #Add inverse edge
        G.add_edge(rel['concept_id_2'],
                   rel['concept_id_1'],
                   id=j + 1,
                   title=inv_rel_name,
                   rel_name=inv_rel_name,
                   rel_type=rel_type_map[inv_rel_name])

        return G, rel_type_map


def create_knowledge_graph(db_conn_string, graph_file, use_cache=True):
    graph_name = Path(graph_file).stem
    rel_file_name = Path(graph_file).parent / f'rel_{graph_name}.json'
    if use_cache:
        if Path(graph_file).exists() and Path(rel_file_name).exists():
            return nx.read_gpickle(graph_file)

    # Create engine and calculate relationships
    engine = create_engine(db_conn_string)
    rels = pd.read_sql_query(all_relationships_query, engine)

    # Build networkX graph
    knowledge_graph, rel_type_map = build_nx_graph(rels)

    # Write graph and relationship map to files
    nx.write_gpickle(knowledge_graph, graph_file)

    with open(Path(graph_file).parent / f'rel_{graph_name}.json', 'w') as f:
        json.dump(rel_type_map, f)

    return knowledge_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Create a graph of drug adverse event and indication information.")
    parser.add_argument(
        'conn_string',
        help='SQL-alchemy compatible database connection string.')
    parser.add_argument('graph_file',
                        help='Output path to save NetworkX graph')

    args = parser.parse_args()

    create_knowledge_graph(args.conn_string, args.graph_file)