'''
Code Overview:
- FastAPI is used to set up a web server and define API endpoints for querying and exporting data.
- On application startup, synthetic data is generated and stored in a PostgreSQL database (spun up within a TestContainer environment).
- SQLModel is used to define the schema of the database ('Observation') and interact with it.
- The 'Observation' class defines the database structure, while the 'Transient' class contains methods to generate synthetic data for populating the database.
- The API provides endpoints to query the database based on a flux threshold and to export filtered observations to an HDF5 file.
- The synthetic data generation is based on Josh's Synthetic Transient Repo (https://github.com/JBorrow/synthetic-transients).
'''

# Import all necessary libraries
from fastapi import FastAPI, Query
from sqlmodel import SQLModel, Field, Session, create_engine, select
from testcontainers.postgres import PostgresContainer
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import List
import random
import numpy as np
import pandas as pd
import math


class Observation(SQLModel, table=True):
    """
    Define the schema of the database table used for storing astronomical observations.
    Each instance of this class represents a single observation with attributes like source name, coordinates (RA, Dec), observation time, and flux measurements at various frequencies.
    """
    id: int | None = Field(default=None, primary_key=True)
    source: str = Field(index=True)
    ra: float
    dec: float
    observation_time: datetime = Field(index=True)

    flux_027: float
    uncertainty_027: float

    flux_039: float
    uncertainty_039: float

    flux_093: float
    uncertainty_093: float

    flux_145: float
    uncertainty_145: float

    flux_225: float
    uncertainty_225: float

    flux_280: float
    uncertainty_280: float


class Transient(BaseModel):
    """
    Define properties of a transient source.
    This function contains methods to generate synthetic observations for it.
    """
    source: str
    ra: float
    dec: float
    index: float  # Frequency-dependence index
    time: timedelta  # Time since peak flux
    peak_flux_093: float  # Peak flux at 93 GHz
    noise_floor: float  # Typical noise floor
    duration: timedelta  # Duration of the transient

    def get_flux(self, frequency: int, flux_093: float) -> float:
        """
        Return the flux at a given frequency.
        """
        return float(
            (flux_093 * (frequency / 93) ** self.index)
            + self.get_uncertainty()
            + self.noise_floor
        )

    def get_uncertainty(self) -> float:
        """
        Return the uncertainty at a given frequency.
        """
        return float(random.random() * math.sqrt(self.noise_floor))

    def get_observations(self, n: int = 365) -> list[Observation]:
        """
        Generate a list of daily observations for the transient.
        """
        START_TIME = datetime.now(timezone.utc)
        TRANSIENT_TIME = START_TIME - self.time

        datetimes = [
            START_TIME - timedelta(days=i) + timedelta(hours=6) * random.random()
            for i in range(n)
        ]

        time_offsets = [(t - TRANSIENT_TIME) / self.duration for t in datetimes]

        fluxes_093 = [
            self.peak_flux_093 * np.exp(-time_offset * time_offset)
            for time_offset in time_offsets
        ]

        observations = []

        for t, flux_093 in zip(datetimes, fluxes_093):
        
            observations.append(
                Observation(
                    source=self.source,
                    ra=self.ra,
                    dec=self.dec,
                    observation_time=t,
                    flux_027=self.get_flux(27, flux_093),
                    uncertainty_027=self.get_uncertainty(),
                    flux_039=self.get_flux(39, flux_093),
                    uncertainty_039=self.get_uncertainty(),
                    flux_093=self.get_flux(93, flux_093),
                    uncertainty_093=self.get_uncertainty(),
                    flux_145=self.get_flux(145, flux_093),
                    uncertainty_145=self.get_uncertainty(),
                    flux_225=self.get_flux(225, flux_093),
                    uncertainty_225=self.get_uncertainty(),
                    flux_280=self.get_flux(280, flux_093),
                    uncertainty_280=self.get_uncertainty(),
                )
            )

        return observations


def synthcat(n: int, engine):
    """
    Populate the database with synthetic observations.
    """
    with Session(engine) as session:
        # Generate synthetic transient data
        transients = [
            Transient(
                source=f"source_{i}",
                ra=random.uniform(0, 360),
                dec=random.uniform(-90, 90),
                index=random.uniform(-2.0, 2.0),
                time=timedelta(days=random.uniform(-1000, 1000)),
                peak_flux_093=random.uniform(0.0, 3.0),
                noise_floor=random.uniform(0.1, 0.5),
                duration=timedelta(days=random.uniform(0, 20)),
            )
            for i in range(n)
        ]

        # Generate observations for each transient
        obs_list = []
        
        for transient in transients:
        
            obs_list.extend(transient.get_observations()) 
        
        session.add_all(obs_list)
        session.commit()

    print(f"Created a database with {n} synthetic observations.")


def query_service(engine, flux_threshold):
    """
    Query the database to retrieve all observations with flux values above a specified threshold.
    """
    with Session(engine) as session:
        
        query = select(Observation).where(Observation.flux_093 > flux_threshold)
        observations = session.exec(query).all()
        
    return observations
    

def export_to_hdf5(observations, hdf5_file):
    """
    Convert a list of queried observations into a pandas dataframe and write the data to an HDF5 file.
    """
    data = [
        {
            "source": obs.source,
            "ra": obs.ra,
            "dec": obs.dec,
            "observation_time": obs.observation_time,
            "flux_027": obs.flux_027,
            "uncertainty_027": obs.uncertainty_027,
            "flux_039": obs.flux_039,
            "uncertainty_039": obs.uncertainty_039,
            "flux_093": obs.flux_093,
            "uncertainty_093": obs.uncertainty_093,
            "flux_145": obs.flux_145,
            "uncertainty_145": obs.uncertainty_145,
            "flux_225": obs.flux_225,
            "uncertainty_225": obs.uncertainty_225,
            "flux_280": obs.flux_280,
            "uncertainty_280": obs.uncertainty_280
        }
        for obs in observations
    ]
    df = pd.DataFrame(data)

    with pd.HDFStore(hdf5_file, mode="w") as store:
        store.put("observations", df, format="table", data_columns=True)

# define the main entry point for defining routes and handling requests
app = FastAPI()

# spin a postgresql database within testcontainer
postgres_container = PostgresContainer("postgres:16")
postgres_container.start()

# initate the postgresql engine 
engine = create_engine(postgres_container.get_connection_url())

# create the database schema
SQLModel.metadata.create_all(engine)

@app.get("/")
def root():
    """
    root endpont for the application.
    """
    return {"message": "Welcome to the Data Query Service"}

# generate synthetic data on api startup
@app.on_event("startup")
async def startup_event():
    # generate 100 synthetic observations
    synthcat(100, engine)  


@app.get("/observations/", response_model=List[Observation])
def get_observations(flux_threshold: float = Query(3.0, description="Flux threshold for filtering observations")):
    """
    Retrieve observations based on a flux threshold. 
    """
    observations = query_service(engine, flux_threshold)
    return observations


@app.get("/export/")
def export_observations(flux_threshold: float = Query(3.0, description="Flux threshold for filtering"), hdf5_file: str = "filtered.h5"):
    """
    Export filtered observations to an HDF5 file.
    """
    observations = query_service(engine, flux_threshold)
    export_to_hdf5(observations, hdf5_file)
    return {"message": f"Filtered data exported to {hdf5_file}"}
