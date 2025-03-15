import os
import sys
import subprocess
import time
import argparse

def run_command(command, description):
    """
    Run a command and print its output
    
    Args:
        command: Command to run
        description: Description of the command
    
    Returns:
        True if the command succeeded, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print(f"{'=' * 80}\n")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed successfully.")
            return True
        else:
            print(f"\n❌ {description} failed with return code {process.returncode}.")
            return False
    
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def check_requirements():
    """
    Check if all required packages are installed
    
    Returns:
        True if all requirements are met, False otherwise
    """
    print("\nChecking requirements...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        # Create requirements.txt
        with open('requirements.txt', 'w') as f:
            f.write("""
tensorflow>=2.5.0
opencv-python>=4.5.0
mediapipe>=0.8.9
numpy>=1.19.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
pandas>=1.3.0
seaborn>=0.11.0
flask>=2.0.0
flask-socketio>=5.0.0
tensorflowjs>=3.8.0
""")
        print("Created requirements.txt")
    
    # Install requirements
    return run_command(
        "pip install -r requirements.txt",
        "Installing required packages"
    )

def prepare_data(args):
    """
    Prepare the dataset
    
    Args:
        args: Command line arguments
    
    Returns:
        True if data preparation succeeded, False otherwise
    """
    return run_command(
        f"python data_preparation.py {' --no-visualize' if args.no_visualize else ''}",
        "Preparing and organizing the dataset"
    )

def train_model(args):
    """
    Train the model
    
    Args:
        args: Command line arguments
    
    Returns:
        True if model training succeeded, False otherwise
    """
    # Create a custom command with the specified arguments
    command = "python train_model.py"
    
    if args.epochs:
        command += f" --epochs {args.epochs}"
    
    if args.batch_size:
        command += f" --batch_size {args.batch_size}"
    
    if args.no_finetune:
        command += " --no-finetune"
    
    return run_command(
        command,
        "Training the ASL recognition model"
    )

def convert_model():
    """
    Convert the model to TensorFlow.js format
    
    Returns:
        True if model conversion succeeded, False otherwise
    """
    return run_command(
        "python convert_model.py",
        "Converting the model to TensorFlow.js format"
    )

def update_backend():
    """
    Update the backend to use the trained model
    
    Returns:
        True if backend update succeeded, False otherwise
    """
    return run_command(
        "python update_backend.py",
        "Updating the backend to use the trained model"
    )

def update_frontend():
    """
    Update the frontend to work with the trained model
    
    Returns:
        True if frontend update succeeded, False otherwise
    """
    return run_command(
        "python update_frontend.py",
        "Updating the frontend to work with the trained model"
    )

def main():
    """
    Main function to run the entire pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the ASL recognition pipeline')
    parser.add_argument('--skip-data-prep', action='store_true', help='Skip data preparation step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip-conversion', action='store_true', help='Skip model conversion step')
    parser.add_argument('--skip-backend', action='store_true', help='Skip backend update step')
    parser.add_argument('--skip-frontend', action='store_true', help='Skip frontend update step')
    parser.add_argument('--no-visualize', action='store_true', help='Skip dataset visualization')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--no-finetune', action='store_true', help='Skip fine-tuning step')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("ASL RECOGNITION PIPELINE".center(80))
    print("=" * 80 + "\n")
    
    # Start time
    start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Failed to install required packages. Exiting.")
        return False
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../dataset', exist_ok=True)
    os.makedirs('../public/models/tfjs_model', exist_ok=True)
    
    # Run pipeline steps
    steps = [
        (not args.skip_data_prep, prepare_data, "Data preparation", args),
        (not args.skip_training, train_model, "Model training", args),
        (not args.skip_conversion, convert_model, "Model conversion", None),
        (not args.skip_backend, update_backend, "Backend update", None),
        (not args.skip_frontend, update_frontend, "Frontend update", None)
    ]
    
    success = True
    for should_run, func, step_name, func_args in steps:
        if should_run:
            if func_args:
                step_success = func(func_args)
            else:
                step_success = func()
            
            if not step_success:
                print(f"\n❌ {step_name} failed. Pipeline may be incomplete.")
                success = False
                if step_name in ["Data preparation", "Model training"]:
                    print("Critical step failed. Exiting pipeline.")
                    break
        else:
            print(f"\n⏩ Skipping {step_name} step.")
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY".center(80))
    print("=" * 80)
    
    if success:
        print("\n✅ ASL recognition pipeline completed successfully!")
    else:
        print("\n⚠️ ASL recognition pipeline completed with some issues.")
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Run the Flask application: python app.py")
    print("2. Open your browser and navigate to http://localhost:5000")
    print("3. Use the application to recognize ASL gestures")
    
    return success

if __name__ == "__main__":
    main()
