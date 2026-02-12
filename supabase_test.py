from dotenv import load_dotenv
from supabase import create_client
import os

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

print(supabase.table("documents").select("*").execute())