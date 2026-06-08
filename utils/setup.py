import os
import warnings
from dotenv import load_dotenv

def initialize_environment():
    warnings.filterwarnings('ignore')

    print(f"Τρέχων φάκελος εργασίας: {os.getcwd()}")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        print("Το HuggingFace Token φορτώθηκε επιτυχώς.")
    else:
        print("Προειδοποίηση: Δεν βρέθηκε HF_TOKEN στο αρχείο .env")
        
    return hf_token