import asyncio
import os
from dotenv import load_dotenv
from adk_workflow import run_adk_workflow, display_results

# Load environment variables
load_dotenv()

async def main():
    """
    Main execution function for the multi-agent system.
    """
    
    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "SERPAPI_API_KEY", "OPENAI_API_KEY"]  # ✅ Added OpenAI
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease check your .env file and ensure all API keys are set.")
        return
    
    # Define the initial search query
    initial_query = input("Enter your search query (or press Enter for default): ").strip()
    
    if not initial_query:
        initial_query = "latest developments in artificial intelligence and machine learning"
    
    print(f"\n🚀 Initializing multi-agent system...")
    print(f"🎯 Search Query: {initial_query}")
    print("⏳ This may take a few moments...\n")
    
    try:
        # Run the ADK workflow
        results = await run_adk_workflow(initial_query)
        
        # Display the results
        display_results(results)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("Please check your API keys and network connection.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
