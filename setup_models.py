"""
Quick setup script to get ExoHunter running
"""

print("🚀 ExoHunter Setup")
print("=================")
print()
print("The trained models are large files (>100MB) and not stored in Git.")
print("To get the models, run:")
print()
print("   python train_exoplanet_models.py")
print()
print("This will train and save the models needed for prediction.")
print("Training takes about 5-10 minutes.")
print()
print("After training, you can run:")
print("   streamlit run streamlit_app.py")
print("   # or")
print("   uvicorn web.api.main:app --reload")