from pydantic import BaseModel

class LoanApplication(BaseModel):
    Income: float
    Credit_Score: float
    Loan_Amount: float
    DTI_Ratio: float
    Employment_Status: str


