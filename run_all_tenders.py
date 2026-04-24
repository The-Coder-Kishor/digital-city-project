import subprocess
import sys

def main():
    # Explicit list of tender IDs
    tender_ids = [
        "632365", "632366", "632368", "632369", "632370", "632372", "632373",
        "632374", "632375", "632377", "632380", "632647", "632666", "632668",
        "632669", "632678", "632684", "632700", "632701", "632752", "632754",
        "632808", "632842", "632855", "632856", "632861", "632871", "632882",
        "632890", "632904", "632931", "632945", "632963", "632977", "632995",
        "633012", "633037", "633113", "633162", "633182", "633183", "633185",
        "633188"
    ]
    
    print(f"Found {len(tender_ids)} tenders in the list. Starting pipeline...")
    
    for tender_id in tender_ids:
        print(f"\n{'='*50}")
        print(f"Processing Tender ID: {tender_id}")
        print(f"{'='*50}")
        
        # Run the python script for the current tender_id
        result = subprocess.run([
            sys.executable, 
            "scripts/agentic_tender_report_pipeline.py", 
            "--tender-id", 
            tender_id
        ])
        
        if result.returncode != 0:
            print(f"Warning: Pipeline for tender {tender_id} exited with status {result.returncode}.")

if __name__ == "__main__":
    main()
