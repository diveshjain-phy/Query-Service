"""
Code Overview:
- FastAPI is used to set up a web server and define API endpoints for querying and exporting data.
- On application startup, synthetic data is generated and stored in a PostgreSQL database (spun up within a TestContainer environment).
- SQLModel is used to define the schema of the database ('Observation') and interact with it.
- The 'Observation' class defines the database structure, while the 'Transient' class contains methods to generate synthetic data for populating the database.
- The API provides endpoints to query the database based on a flux threshold and to export filtered observations to an HDF5 file.
- The synthetic data generation is based on Josh's Synthetic Transient Repo (https://github.com/JBorrow/synthetic-transients).
"""

# Import all necessary libraries
from fastapi import FastAPI, Depends, Query
from sqlmodel import SQLModel, Field, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from testcontainers.postgres import PostgresContainer
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
import random
import numpy as np
import math
import io
import h5py
from fastapi.responses import StreamingResponse
from sqlalchemy import Column, TIMESTAMP


class Observation(SQLModel, table=True):
    """
    Define the schema of the database table used for storing astronomical observations.
    Each instance of this class represents a single observation with attributes like source name,
    coordinates (RA, Dec), observation time, and flux measurements at various frequencies.
    """

    id: int | None = Field(default=None, primary_key=True)
    source: str = Field(index=True)
    ra: float
    dec: float
    observation_time: datetime = Field(sa_column=Column(TIMESTAMP(timezone=True)))

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
                    observation_time=t.astimezone(timezone.utc),
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


async def synthcat(n: int, session: AsyncSession):
    """
    Populate the database with synthetic observations.
    """
    # Generate synthetic transient data
    transients = [
        Transient(
            source=f'source_{i}',
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
    await session.commit()

    print(f'Created a database with {n} synthetic observations.')


async def query_service(session: AsyncSession, flux_threshold: float):
    """
    Query the database to retrieve all observations with flux values above a specified threshold.
    """
    query = select(Observation).where(Observation.flux_093 > flux_threshold)
    result = await session.execute(query)
    return result.scalars().all()


def export_to_hdf5(observations):
    """
    Convert queried observations directly into an HDF5 file in memory using lists.
    """
    # Prepare data as lists
    sources = [obs.source for obs in observations]  # Strings
    ras = [obs.ra for obs in observations]
    decs = [obs.dec for obs in observations]
    observation_times = [str(obs.observation_time) for obs in observations]
    flux_027s = [obs.flux_027 for obs in observations]
    uncertainty_027s = [obs.uncertainty_027 for obs in observations]
    flux_039s = [obs.flux_039 for obs in observations]
    uncertainty_039s = [obs.uncertainty_039 for obs in observations]
    flux_093s = [obs.flux_093 for obs in observations]
    uncertainty_093s = [obs.uncertainty_093 for obs in observations]
    flux_145s = [obs.flux_145 for obs in observations]
    uncertainty_145s = [obs.uncertainty_145 for obs in observations]
    flux_225s = [obs.flux_225 for obs in observations]
    uncertainty_225s = [obs.uncertainty_225 for obs in observations]
    flux_280s = [obs.flux_280 for obs in observations]
    uncertainty_280s = [obs.uncertainty_280 for obs in observations]

    # Use BytesIO to create an in-memory binary stream
    hdf5_buffer = io.BytesIO()

    # Use h5py to write to the in-memory buffer
    with h5py.File(hdf5_buffer, 'w') as hdf5_file:
        group = hdf5_file.create_group('observations')
        group.create_dataset('source', data=sources, dtype=h5py.special_dtype(vlen=str))
        group.create_dataset('ra', data=ras)
        group.create_dataset('dec', data=decs)
        group.create_dataset(
            'observation_time', data=observation_times, dtype=h5py.special_dtype(vlen=str)
        )
        group.create_dataset('flux_027', data=flux_027s)
        group.create_dataset('uncertainty_027', data=uncertainty_027s)
        group.create_dataset('flux_039', data=flux_039s)
        group.create_dataset('uncertainty_039', data=uncertainty_039s)
        group.create_dataset('flux_093', data=flux_093s)
        group.create_dataset('uncertainty_093', data=uncertainty_093s)
        group.create_dataset('flux_145', data=flux_145s)
        group.create_dataset('uncertainty_145', data=uncertainty_145s)
        group.create_dataset('flux_225', data=flux_225s)
        group.create_dataset('uncertainty_225', data=uncertainty_225s)
        group.create_dataset('flux_280', data=flux_280s)
        group.create_dataset('uncertainty_280', data=uncertainty_280s)

    # Reset the buffer's position to the start
    hdf5_buffer.seek(0)

    return hdf5_buffer


async def init_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


# define a dependency for the database session
async def get_session():
    async with Session() as session:
        yield session


# spin a postgresql database within testcontainer
postgres_container = PostgresContainer('postgres:16')
postgres_container.start()

# initate the postgresql engine
db_url = postgres_container.get_connection_url()
db_async_url = db_url.replace('postgresql+psycopg2://', 'postgresql+asyncpg://')
engine = create_async_engine(db_async_url)
Session = async_sessionmaker(engine, expire_on_commit=False)


# generate synthetic data on api startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Create the database schema
        await init_db(engine)

        # Generate synthetic data
        async with Session() as session:
            await synthcat(100, session)

        yield
    except Exception as e:
        print(f'Exception during startup: {e}')
        raise
    finally:
        postgres_container.stop()


# define the main entry point for defining routes and handling requests
app = FastAPI(lifespan=lifespan)


@app.get('/')
def root():
    """
    root endpont for the application.
    """
    return {'message': 'Welcome to the Data Query Service'}


@app.get('/observations/', response_model=List[Observation])
async def get_observations(
    flux_threshold: float = Query(
        3.0, description='Flux threshold for filtering observations'
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve observations based on a flux threshold.
    """
    observations = await query_service(session, flux_threshold)
    return observations


@app.get('/export/')
async def export_observations(
    flux_threshold: float = Query(3.0, description='Flux threshold for filtering'),
    session: AsyncSession = Depends(get_session),
):
    observations = await query_service(session, flux_threshold)
    hdf5_buffer = export_to_hdf5(observations)

    return StreamingResponse(
        hdf5_buffer,
        media_type='application/x-hdf5',
        headers={'Content-Disposition': 'attachment; filename=filtered_data.h5'},
    )
