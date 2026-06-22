"""OCR system diagnostic script."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

print('=== OCR SYSTEM DIAGNOSTIC ===\n')

# 1. Check EasyOCR installation
print('1. Checking EasyOCR installation...')
try:
    import easyocr
    print(f'   OK: easyocr installed')
except ImportError as e:
    print(f'   ERROR: easyocr not installed - {e}')
    sys.exit(1)

# 2. Check ocr_validator module
print('\n2. Checking ocr_validator module...')
try:
    from ocr_validator import validate_container_number, _get_reader
    print('   OK: ocr_validator imported successfully')
except ImportError as e:
    print(f'   ERROR: Cannot import ocr_validator - {e}')
    sys.exit(1)

# 3. Test EasyOCR Reader initialization
print('\n3. Testing EasyOCR Reader initialization...')
try:
    reader = _get_reader()
    if reader is None:
        print('   WARNING: EasyOCR Reader is None (will use passthrough mode)')
    else:
        print('   OK: EasyOCR Reader loaded successfully')
except Exception as e:
    print(f'   ERROR: Failed to initialize reader - {e}')
    sys.exit(1)

# 4. Test OCR validator with a test image
print('\n4. Testing OCR validator with synthetic image...')
try:
    # Create a simple test image with text
    test_img = Image.new('RGB', (200, 100), color='white')
    d = ImageDraw.Draw(test_img)
    d.text((10, 40), 'MSCU1234567', fill='black')

    result = validate_container_number(test_img)
    print(f'   Found: {result["found"]}')
    print(f'   Number: {result["number"]}')
    print(f'   Confidence: {result["confidence"]}')
    print(f'   Is marking panel: {result["is_marking_panel"]}')
    print(f'   Message: {result["message"]}')
    if result['found']:
        print('   OK: OCR validator working')
    else:
        print('   WARNING: No text detected in test image')
except Exception as e:
    print(f'   ERROR: OCR validation failed - {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Check ContainerPredictor OCR integration
print('\n5. Testing ContainerPredictor OCR integration...')
try:
    if not Path('models/container_classifier.pth').exists():
        print('   SKIP: Model not found (run train.py first)')
    else:
        from inference import ContainerPredictor
        predictor = ContainerPredictor()
        if predictor._ocr_validate is not None:
            print('   OK: OCR validator integrated into ContainerPredictor')
        else:
            print('   ERROR: OCR validator not loaded in ContainerPredictor')
except Exception as e:
    print(f'   ERROR: Failed to test ContainerPredictor - {e}')
    import traceback
    traceback.print_exc()

print('\n=== DIAGNOSTIC COMPLETE ===')
