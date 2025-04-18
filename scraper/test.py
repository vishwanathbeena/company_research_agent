from langchain_core.pydantic_v1 import BaseModel,Field
from typing import List, Dict, Set, Any, Optional

class PersonName(BaseModel):
    Name:str = Field(description="Name of the person")
    Title:str = Field(description="Title of the person in the company")

class CompanyFinancials(BaseModel):
    Revenue: str = Field(description="Revenue of the company or Turnover")
    Profit: str = Field(description="Profit of the company")
    Loss: str = Field(description="Loss of the company")
    Financial_Year:str = Field(description="Financial Year of the company")

class CompanyTaxDetails(BaseModel):
    Tax_number:str = Field(description="Tax Number of the company like GSTIN,ITIN,PAN,EIN etc.")
    Tax_Type:str = Field(description="Tax Type of the company like GST,IT,PAN etc.")
    State:str = Field(description="State of the company")
    Country:str = Field(description="Country of the company")

class CompanyContactInformation(BaseModel):
    Email:str = Field(description="Email of the company")
    Phone:str = Field(description="Phone of the company")
    Address:str = Field(description="Address of the company")
    Website:str = Field(description="Website of the company")

class CommpanyAssociatedOrganizations(BaseModel):
    Organization_Name:str = Field(description="Organization Name")
    Organization_Type:str = Field(description="Organization Type")
    Relationship:str = Field(description="Relationship with the company")

class CompanyProfile(BaseModel):
    Company_Name: str = Field(description="Company Name")
    Description: str = Field(description="Description about the company")
    Key_People: List[PersonName] = Field(description="Key People in the company")
    # Number_of_Employees:str = Field(description="Number of Employees in the company")
    # Major_Projects_or_Products:List[str] = Field(description="Major Projects or Products of the company")
    # Financial_Information:List[CompanyFinancials] = Field(description="Financial Information of the company")
    # Business_Model:str = Field(description="Business Model of the company")
    # Founding_Date:str = Field(description="Founding Date of the company")
    # Industry_Sector:str = Field(description="Industry Sector of the company")
    # Headquarters_Location:str = Field(description="Headquarters Location of the company")
    # Company_Structure:str = Field(description="Company Structure")
    # Company_Contact_Information:List[CompanyContactInformation ] = Field(description="Company Contact Information")
    # Key_Competitors:List[str] = Field(description="Key Competitors of the company")
    # Geographies_Served:List[str] = Field(description="Geographies Served by the company")
    # Number_of_Branches:str = Field(description="Number of Branches of the company")
    # Tax_Details:List[CompanyTaxDetails] = Field(description="Tax Details of the company")
    # Company_Registration_Details:str = Field(description="Company Registration Details of the company")
    # Company_Compliance_Details:str = Field(description="Company Compliance Details of the company")
    # Associated_Organizations:List[CommpanyAssociatedOrganizations] = Field(description="Associated Organizations of the company")
    # Sources:List[str] = Field(description="Sources of the information")

