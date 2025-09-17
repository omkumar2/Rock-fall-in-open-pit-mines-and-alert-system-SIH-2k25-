import ee

print("Running Earth Engine authentication. This will open a browser window for you to log in.")
print("Please follow the instructions in the browser and paste the authorization code back here if prompted.")

try:
    ee.Authenticate()
    ee.Initialize()
    print("\nEarth Engine authentication and initialization successful!")
    print("You can now use GEE functions in your scripts.")
except Exception as e:
    print(f"\nError during Earth Engine authentication or initialization: {e}")
    print("Please ensure you have an active Google Cloud project and have enabled the Earth Engine API.")
    print("You might need to run 'earthengine authenticate --quiet' or 'earthengine set_project <your-project-id>' manually.")
