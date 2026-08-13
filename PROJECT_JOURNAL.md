### Project Direction

EmberRisk focuses on wildfire-risk classification using heterogeneous public environmental data.

The project uses historical wildfire detections and weather data to build a daily grid-based dataset for California.

### Current Status

- Repository initialized
- Python virtual environment created
- Phase 1 data source study completed
- Phase 2 ingestion architecture completed
- FIRMS data ingested with manifest-based resumability
- NASA POWER data ingested for the modeling period
- Phase 3 processing completed
- 323 canonical grid cells
- 943,806 cell-day records generated
- Fire-history features generated using the warm-up period
- 3, 7 and 14-day targets generated
- Phase 3 validation completed successfully
- 74 automated tests passing

### Phase 3 Output

The processed dataset is stored in:

`data/processed/cell_day_dataset.parquet`

Targets are stored separately for 3, 7 and 14-day horizons.

Validation results are stored in:

`data/processed/validation_report.json`

### Next Step

Begin Phase 4: exploratory data analysis and feature analysis.
