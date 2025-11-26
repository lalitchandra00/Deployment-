from pydantic import BaseModel

class Values(BaseModel):
    BHK: int
    Size: int
    Area: str
    City: str
    Furnishing: str
    Bathroom: int