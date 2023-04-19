# Python imports
import os
import logging
import argparse
import configparser

# Third-party imports
from pymysql.constants import CLIENT
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
import pandas as pd

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

USE_SSH_TUNNEL = True

# table names
COHORT_TABLE = "cohort_person"
COHORT_DRUG_ERA_TABLE = "cohort_drug_era"
ADVERSE_EVENT_TABLE = "cohort_adverse_event"
CONCEPT_SEQUENCES_TABLE = "cohort_concept_sequence"

OMOP_SCHEMA = CONFIG['DATABASE']['omop_schema']
COHORT_SCHEMA = CONFIG['DATABASE']['cohort_schema']


def connect_to_server():
    if USE_SSH_TUNNEL:
        server = SSHTunnelForwarder(
            (CONFIG['DATABASE']['ssh_server'], 22),
            ssh_password=CONFIG['DATABASE']['ssh_pass'],
            ssh_username=CONFIG['DATABASE']['ssh_user'],
            remote_bind_address=('127.0.0.1', 3306))

        server.start()

    if USE_SSH_TUNNEL:
        engine = create_engine(
            f"mysql+pymysql://{CONFIG['DATABASE']['my_sql_user']}:{CONFIG['DATABASE']['my_sql_pass']}@{CONFIG['DATABASE']['my_sql_host']}:{server.local_bind_port}/{CONFIG['DATABASE']['my_sql_db']}",
            connect_args={"client_flag": CLIENT.MULTI_STATEMENTS})
    else:
        engine = create_engine(
            f"mysql+pymysql://{CONFIG['DATABASE']['my_sql_user']}:{CONFIG['DATABASE']['my_sql_pass']}@{CONFIG['DATABASE']['my_sql_host']}/{CONFIG['DATABASE']['my_sql_db']}",
            connect_args={"client_flag": CLIENT.MULTI_STATEMENTS})

    connection = engine.connect()
    return connection


DRUG_ERA_QUERY = """
	SELECT
		a.*, (
					CASE person_id
					WHEN @cur_person_id
					THEN @curRow := @curRow + 1
					ELSE @curRow := 1 AND @cur_person_id := person_id END
			) AS rank
	FROM (
		SELECT
			de.person_id, de.drug_era_id,
			de.drug_concept_id, drug_concept.concept_name drug_concept_name, drug_era_start_date, drug_era_end_date
		FROM {omop_schema}.drug_era de
		JOIN {omop_schema}.concept drug_concept
			ON drug_concept.concept_id = de.drug_concept_id
		WHERE de.drug_concept_id IN ({drug_concepts})

	) a, (SELECT @curRow := 0, @cur_person_id := -1) R
	ORDER BY person_id, drug_era_start_date DESC

"""

ADVERSE_EVENT_QUERY = """
SELECT DISTINCT
	de.person_id, drug_era_id,
    de.drug_concept_id, meddra_c.concept_id meddra_condition_concept_id
FROM {cohort_schema}.cohort_drug_era de
JOIN {omop_schema}.condition_occurrence ce
	ON ce.person_id = de.person_id
		AND ce.condition_start_date BETWEEN de.drug_era_start_date AND DATE_ADD(de.drug_era_start_date, INTERVAL 52 WEEK)
JOIN {omop_schema}.concept_ancestor ca
	ON ca.descendant_concept_id = ce.condition_concept_id
JOIN {omop_schema}.concept_relationship cr
	ON cr.concept_id_1 = ca.ancestor_concept_id
		AND cr.relationship_id = 'SNOMED - MedDRA eq'
JOIN {omop_schema}.concept meddra_c
	ON meddra_c.concept_id = cr.concept_id_2
		AND meddra_c.vocabulary_id = 'MedDRA'
WHERE meddra_c.concept_id IN (
    {ae_concept_ids}
    );
"""

CONCEPT_SEQUENCES_QUERY = """
        SELECT
            person_id, event_id, union_concept.concept_id, c.concept_name, concept_date, domain_id
        FROM (
            SELECT
                co.person_id, condition_era_id event_id, condition_concept_id concept_id, condition_era_start_date concept_date,
                0 is_index
            FROM {omop_schema}.condition_era co
            JOIN {cohort_schema}.cohort_drug_era p
                ON p.person_id = co.person_id
                    AND rank = 1
            WHERE co.condition_era_start_date <= p.drug_era_start_date
            UNION ALL
            SELECT
                de.person_id, de.drug_era_id event_id, de.drug_concept_id concept_id, de.drug_era_start_date concept_date,
                0 is_index
            FROM {omop_schema}.drug_era de
            JOIN {cohort_schema}.cohort_drug_era p
                ON p.person_id = de.person_id
                    AND rank = 1
            WHERE de.drug_era_start_date <= p.drug_era_start_date
            UNION ALL
            SELECT
                m.person_id, measurement_id event_id, measurement_concept_id concept_id, m.measurement_date concept_date,
                0 is_index
            FROM {omop_schema}.measurement m
            JOIN {cohort_schema}.cohort_drug_era p
                ON p.person_id = m.person_id
                    AND rank = 1
            WHERE m.measurement_date <= p.drug_era_start_date
                AND (m.value_as_number < range_low OR m.value_as_number > range_high)
            # UNION ALL
            # SELECT
            #     de.person_id, p4.drug_era_id, de.drug_era_id event_id, drug_concept_id concept_id, de.drug_era_start_date concept_date,
            #     1 is_index
            # FROM {omop_schema}.drug_era de
            # JOIN {cohort_schema}.cohort_drug_era p
            #     ON p.person_id = de.person_id
            #         AND p.drug_era_id = de.drug_era_id
        ) union_concept
        JOIN {omop_schema}.concept c
            ON c.concept_id = union_concept.concept_id
        ORDER BY person_id, concept_date;
"""

CONCEPT_SEQUENCES_WITH_DRUG_ERAS_QUERY = """
SELECT
	person_id, drug_era_id, drug_concept_id, event_id, concept_id,
    concept_name, concept_date, domain_id, CONVERT(rank, SIGNED) rank
FROM (
	
    SELECT
		base_rows.*, @rank := 
				CASE  
					WHEN @partval = drug_era_id AND @rankval1 = concept_date AND @rankval2 = event_id THEN @rank
					WHEN @partval = drug_era_id AND (@rankval1 := concept_date) IS NOT NULL
						AND (@rankval2 := event_id) IS NOT NULL THEN @rank + 1
					WHEN (@partval := drug_era_id) IS NOT NULL 
						AND (@rankval1 := concept_date) IS NOT NULL AND (@rankval2 := event_id) THEN 1
					ELSE @rank := @rank + 1
				END AS rank
	FROM (
	SELECT
		de.person_id, de.drug_era_id, de.drug_concept_id, cs.event_id,
        cs.concept_id, cs.concept_name, cs.concept_date, cs.domain_id
	FROM {cohort_schema}.cohort_drug_era de
	JOIN {cohort_schema}.cohort_concept_sequence cs
		ON de.person_id = cs.person_id
			AND cs.concept_date <= de.drug_era_start_date
	JOIN {cohort_schema}.vocabulary v
		ON v.concept_id = cs.concept_id
    ORDER BY de.drug_era_id, concept_date DESC, event_id ASC

    ) base_rows CROSS JOIN (SELECT @rank := NULL, @partval := NULL, @rankval1 := NULL, @rankval2 := NULL) R

) a
WHERE CONVERT(rank, SIGNED) <151
{optional_where_clause}
ORDER BY drug_era_id, CONVERT(rank, SIGNED) ASC;

"""

ADVERSE_EVENT_LABELS_QUERY = """
SELECT
	de.person_id, de.drug_era_id, de.drug_concept_id, IF(ae.drug_era_id IS NOT NULL, 1, 0) hasDx
FROM {cohort_schema}.cohort_drug_era de
LEFT JOIN {cohort_schema}.cohort_adverse_event ae
	ON ae.drug_era_id = de.drug_era_id
		AND ae.person_id = de.person_id
		AND meddra_condition_concept_id = {adverse_effect_concept_id}
"""

DRUG_ERAS_WITH_DEMOGRAPHICS = """
SELECT
    de.person_id, de.drug_era_id, de.drug_concept_id, hasDx,
    TIMESTAMPDIFF(YEAR, DATE_ADD(DATE_ADD(MAKEDATE(year_of_birth, 1), INTERVAL (month_of_birth)-1 MONTH), INTERVAL (day_of_birth)-1 DAY), de.drug_era_start_date) age,
    gender_code, race_code,
    IF(past_condition_occurrence.person_id IS NOT NULL, 1, 0) pastDx
FROM {cohort_schema}.cohort_drug_era de
JOIN {cohort_schema}.cohort_person cp
    ON cp.person_id = de.person_id
LEFT JOIN (
	SELECT DISTINCT cs.person_id, cs.concept_date,
         ( 
					CASE cs.person_id
					WHEN @cur_person_id
					THEN @curRow := @curRow + 1 
					ELSE @curRow := 1 AND @cur_person_id := cs.person_id END
			) AS rank
	FROM (SELECT @curRow := 0, @cur_person_id := -1) R, {cohort_schema}.cohort_concept_sequence cs 
    JOIN {omop_schema}.concept_ancestor ca
        ON ca.descendant_concept_id = cs.concept_id
    JOIN {omop_schema}.concept_relationship cr
        ON cr.concept_id_1 = ca.ancestor_concept_id
            AND cr.relationship_id = 'SNOMED - MedDRA eq'
    JOIN {omop_schema}.concept meddra_c
        ON meddra_c.concept_id = cr.concept_id_2
            AND meddra_c.vocabulary_id = 'MedDRA'
	WHERE meddra_c.concept_id = {adverse_effect_concept_id}
    ORDER BY cs.person_id, cs.concept_date
) past_condition_occurrence
	ON past_condition_occurrence.person_id = de.person_id
		AND past_condition_occurrence.concept_date < de.drug_era_start_date
			AND past_condition_occurrence.rank = 1
LEFT JOIN (
    SELECT
        de.person_id, de.drug_era_id, IF(ae.drug_era_id IS NOT NULL, 1, 0) hasDx
    FROM {cohort_schema}.cohort_drug_era de
    LEFT JOIN {cohort_schema}.cohort_adverse_event ae
        ON ae.drug_era_id = de.drug_era_id
            AND ae.person_id = de.person_id
            AND meddra_condition_concept_id = {adverse_effect_concept_id}
) future_condition_occurrence
	ON future_condition_occurrence.person_id = de.person_id
		AND future_condition_occurrence.drug_era_id = de.drug_era_id
"""


def retrieve_drug_eras(drug_concept_ids, destfile, desttable):
    conn = connect_to_server()
    conn.execute(f"TRUNCATE TABLE {COHORT_SCHEMA}.{desttable}")

    drug_concept_ids_str = " ,".join(map(str, drug_concept_ids))
    drug_era_query = DRUG_ERA_QUERY.format(drug_concepts=drug_concept_ids_str,
                                           omop_schema=OMOP_SCHEMA,
                                           cohort_schema=COHORT_SCHEMA)

    drug_eras = pd.read_sql_query(drug_era_query, con=conn)
    drug_eras.to_csv(destfile, index=False)
    drug_eras.to_sql(desttable,
                     conn,
                     schema=COHORT_SCHEMA,
                     index=False,
                     if_exists='append')

    return drug_eras


def retrieve_adverse_events(ae_concept_ids, desttable):
    conn = connect_to_server()
    conn.execute(f"TRUNCATE TABLE {COHORT_SCHEMA}.{desttable}")

    ae_concept_ids_str = " ,".join(map(str, ae_concept_ids))
    ae_query = ADVERSE_EVENT_QUERY.format(ae_concept_ids=ae_concept_ids_str,
                                          omop_schema=OMOP_SCHEMA,
                                          cohort_schema=COHORT_SCHEMA)

    adverse_events = pd.read_sql_query(ae_query, con=conn)
    adverse_events.to_sql(desttable,
                          conn,
                          schema=COHORT_SCHEMA,
                          index=False,
                          if_exists='append')

    return adverse_events


def get_concept_sequences(desttable):
    conn = connect_to_server()
    conn.execute(f"TRUNCATE TABLE {COHORT_SCHEMA}.{desttable}")

    concept_sequences_query = CONCEPT_SEQUENCES_QUERY.format(
        omop_schema=OMOP_SCHEMA, cohort_schema=COHORT_SCHEMA)
    concept_sequences = pd.read_sql_query(concept_sequences_query, con=conn)
    concept_sequences.to_sql(desttable,
                             conn,
                             schema=COHORT_SCHEMA,
                             index=False,
                             if_exists='append')

    return concept_sequences


def get_concept_sequences_with_drug_era_ids(drug_era_id=None):
    conn = connect_to_server()

    concept_sequences_with_drug_eras_query = CONCEPT_SEQUENCES_WITH_DRUG_ERAS_QUERY.format(
        omop_schema=OMOP_SCHEMA,
        cohort_schema=COHORT_SCHEMA,
        optional_where_clause=f'AND drug_era_id={drug_era_id}'
        if drug_era_id else '')

    concept_sequences_with_drug_eras = pd.read_sql_query(
        concept_sequences_with_drug_eras_query, conn)

    return concept_sequences_with_drug_eras


def get_adverse_event_labels(adverse_effect_concept_id):
    conn = connect_to_server()
    adverse_event_labels_query = ADVERSE_EVENT_LABELS_QUERY.format(
        adverse_effect_concept_id=adverse_effect_concept_id,
        cohort_schema=COHORT_SCHEMA)
    adverse_event_labels = pd.read_sql_query(adverse_event_labels_query,
                                             con=conn)

    return adverse_event_labels


def get_drug_eras_with_demographics(adverse_effect_concept_id):
    conn = connect_to_server()
    drug_eras_with_demographics_query = DRUG_ERAS_WITH_DEMOGRAPHICS.format(
        adverse_effect_concept_id=adverse_effect_concept_id,
        omop_schema=OMOP_SCHEMA,
        cohort_schema=COHORT_SCHEMA)

    drug_eras_with_demographics = pd.read_sql_query(
        drug_eras_with_demographics_query, con=conn)

    return drug_eras_with_demographics


def write_vocabulary_to_table(knowledge_graph):
    conn = connect_to_server()
    desttable = CONFIG['DATABASE']['vocabulary_table']

    conn.execute(f"TRUNCATE TABLE {COHORT_SCHEMA}.{desttable}")

    rows = [(ndata['concept_id'], ndata['concept_name'], ndata['domain_id'])
            for _, ndata in knowledge_graph.nodes(data=True)]
    df = pd.DataFrame.from_records(
        rows, columns=['concept_id', 'concept_name', 'domain_id'])

    logging.info(f"Writing vocabulary to {COHORT_SCHEMA}.{desttable}")
    df.to_sql(desttable,
              conn,
              schema=COHORT_SCHEMA,
              index=False,
              if_exists='append')


def get_vocabulary_table():
    conn = connect_to_server()
    vocab_table = CONFIG['DATABASE']['vocabulary_table']
    df = pd.read_sql(
        f"""
        SELECT
            v.concept_id, v.concept_name, c.domain_id
        FROM {COHORT_SCHEMA}.{vocab_table} v
        JOIN {OMOP_SCHEMA}.concept c
            ON c.concept_id = v.concept_id
    """, conn)

    return df


def build_cohort(drug_concept_ids, adverse_effect_concept_ids, drug_eras_file):

    logging.info("Retrieving drug eras for cohort")
    drug_eras = retrieve_drug_eras(drug_concept_ids, drug_eras_file,
                                   COHORT_DRUG_ERA_TABLE)
    logging.info(f"{len(drug_eras)} drug eras found")

    logging.info("Retrieving adverse events")
    adverse_events = retrieve_adverse_events(adverse_effect_concept_ids,
                                             ADVERSE_EVENT_TABLE)

    logging.info(f"{len(adverse_events)} found")

    logging.info(f"Getting sequences of concepts")
    concept_sequences = get_concept_sequences(CONCEPT_SEQUENCES_TABLE)

    logging.info("Done.")

    return concept_sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Define a cohort, retrieve its drug eras, and find periods of adverse events."
    )
    parser.add_argument('drug_concept_ids', nargs='+')
    parser.add_argument('adverse_effect_concept_ids', nargs='+')
    parser.add_argument('cohort_file')
    parser.add_argument('adverse_event_file')

    args = parser.parse_args

    build_cohort(args.drug_concept_ids, args.adverse_effect_concept_ids,
                 args.cohort_file, args.adverse_event_file)
