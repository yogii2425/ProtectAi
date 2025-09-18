# backend/db.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Database setup (SQLite file inside project)
DATABASE_URL = "sqlite:///protectai.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Database Model
class Fingerprint(Base):
    __tablename__ = "fingerprints"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    hash_value = Column(String)

# Initialize DB
def init_db():
    Base.metadata.create_all(bind=engine)

# Save fingerprint to DB
def save_fingerprint(filename: str, hash_value: str):
    session = SessionLocal()
    try:
        fp = Fingerprint(filename=filename, hash_value=hash_value)
        session.add(fp)
        session.commit()
    finally:
        session.close()

# Fetch all fingerprints
def get_fingerprints():
    session = SessionLocal()
    try:
        return session.query(Fingerprint).all()
    finally:
        session.close()
