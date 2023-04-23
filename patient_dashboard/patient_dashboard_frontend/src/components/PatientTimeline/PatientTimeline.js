import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import Chart from "react-apexcharts";

const DEFAULT_OPTIONS = {
    chart: {
        type: 'rangeBar'
    },
    plotOptions: {
        bar: {
            horizontal: true,
            barHeight: '70%',
            // distributed: true,
            rangeBarOverlap: true,
            rangeBarGroupRows: false,
            columnWidth: '100%',
            borderRadiusWhenStacked: 'all'
        }
    },
    xaxis: {
        type: 'datetime'
    },
    stroke: {
        width: 3
    },
    fill: {
        type: 'solid',
        opacity: 0.6
    },
    legend: {
        position: 'top',
        horizontalAlign: 'left'
    },

    tooltip: {
        custom: function ({ series, seriesIndex, dataPointIndex, w }) {
            let data = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
            let combined_data = [];

            for (const point of w.globals.initialSeries[seriesIndex].data) {
                if ((data.y[0] <= point.y[1]) && (data.y[1] >= point.y[0])) {
                    combined_data.push(point.concept_name)
                }
            }

            return '<div class="arrow_box">' +
                combined_data.map((data) => '<span>' + data + '</span>').join('<br/>')
                +
                '</div>'
        }
    },
};

export default function PatientTimeline({ data }) {
    const [series, setSeries] = useState([]);
    const [options, setOptions] = useState(DEFAULT_OPTIONS);

    const parseRows = (rows) => {
        let data = rows;
        let series_entries = {}

        data.forEach(row => {
            let concept_name = row['concept_name'];
            let concept_date = row['concept_date'];
            let concept_date_obj = new Date(Date.parse(concept_date));
            let concept_date_plus_one_obj = new Date(Date.parse(concept_date));
            concept_date_plus_one_obj.setDate(concept_date_plus_one_obj.getDate() + 5);

            let x = 'Patient Timeline';
            if (Math.random() < 0.02) {
                x = 'Key Features'
            }

            let domain_id = row['domain_id'];
            if (domain_id in series_entries) {
                series_entries[domain_id].push({
                    x: x,
                    y: [concept_date_obj.getTime(), concept_date_plus_one_obj.getTime()],
                    concept_name: concept_name

                })
            } else {
                series_entries[domain_id] = []
                series_entries[domain_id].push({
                    x: x,
                    y: [concept_date_obj.getTime(), concept_date_plus_one_obj.getTime()],
                    concept_name: concept_name
                })
            }
        });

        let series_entries_list = []

        for (const [k, v] of Object.entries(series_entries)) {
            series_entries_list.push(
                {
                    name: k,
                    data: v
                }
            )
        }

        setSeries(series_entries_list);

    }

    useEffect(() => {
        const fetchRows = async () => {
            parseRows(data);
        }

        fetchRows()

    }, [data]);

    // useEffect(() => {
    //     const fetchRows = async () => {
    //         const response = await fetch('/csv/mini_concept_sequences.csv');
    //         const reader = response.body.getReader();
    //         const result = await reader.read();
    //         const decoder = new TextDecoder('utf-8');
    //         const csv = decoder.decode(result.value);
    //         const results = Papa.parse(csv, { header: true, dynamicTyping: true });
    //         const csv_rows = results.data;
    //         parseRows(csv_rows);
    //     }

    //     fetchRows()

    // }, []);


    return <Chart type="rangeBar" options={options} series={series} height='150' ></Chart>
}