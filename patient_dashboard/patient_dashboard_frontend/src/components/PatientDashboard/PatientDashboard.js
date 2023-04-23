import React, { useState, useEffect } from "react";

import Papa from "papaparse";
import Autocomplete from "@mui/material/Autocomplete";
import { TextField, CircularProgress } from "@mui/material";
import { Container } from "@mui/material";

import PatientTimeline from "../PatientTimeline/PatientTimeline";

export default function PatientDashboard() {
    const [data, setData] = useState([]);
    const [persons, setPersons] = useState([]);
    const [options, setOptions] = useState([]);
    const [person_id, setPersonId] = useState(0);
    const [dataFiltered, setDataFiltered] = useState([]);
    const [isFilteringData, setIsFilteringData] = useState([false])

    useEffect(() => {
        const fetchRows = async () => {
            const response = await fetch('/csv/concept_sequences.csv');
            const reader = response.body.getReader();
            const result = await reader.read();
            const decoder = new TextDecoder('utf-8');
            const csv = decoder.decode(result.value);
            const results = Papa.parse(csv, { header: true, dynamicTyping: true });
            const csv_rows = results.data;

            setData(csv_rows)
        }

        fetchRows()

    }, []);

    useEffect(() => {
        let persons = data.map(row => row.person_id);
        persons = [...new Set(persons)];
        setPersons(persons)
    }, [data]);

    useEffect(() => {
        let options = []
        for (let i in persons) {
            options.push({ 'label': String(persons[i]) })
        }

        setOptions(options);
    }, [persons])

    useEffect(() => {
        setIsFilteringData(true);

        let dataFiltered = data.filter(row => row.person_id == person_id);
        setDataFiltered(dataFiltered);

        setIsFilteringData(false);
    }, [person_id])

    return (
        <Container>
            <h1>Patient Dashboard</h1>
            <Autocomplete
                disablePortal options={options}
                renderInput={(params) => <TextField {...params} label="person_id" />}
                onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                        // event.defaultMuiPrevented = true;
                        console.log(event.target.value);
                        setPersonId(event.target.value);
                    }

                }}
            />
            {isFilteringData && <CircularProgress />}
            {!isFilteringData && dataFiltered.length > 0 &&
                <PatientTimeline data={dataFiltered} />
            }

        </Container>


    )


}