import sys
import traceback

try:
    print("Testing imports...")
    import torch
    import transformers
    import modelscope
    import faiss
    import sentence_transformers
    import jieba
    print("Imports OK.")
    
    print("Testing GPU...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
    print("Starting evaluate.py logic...")
    # Try to import and run main
    sys.path.append('src')
    import evaluate
    evaluate.run_pilot()
    
except Exception as e:
    print("\n!!! CRASH DETECTED !!!")
    traceback.print_exc()
    sys.exit(1)
