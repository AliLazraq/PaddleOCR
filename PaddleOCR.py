"""
Fixed PaddleOCR Text Extraction - Now with PDF Support!
Works with your specific PaddleOCR version format
"""

from paddleocr import PaddleOCR
import cv2
import os
import time
import fitz  # PyMuPDF for PDF processing
import numpy as np

def extract_text_from_result(result):
    """
    Extract text from your specific PaddleOCR result format
    """
    extracted_texts = []
    
    print(f"🔍 Analyzing result structure...")
    print(f"   Result type: {type(result)}")
    print(f"   Result length: {len(result) if result else 'None'}")
    
    if result and len(result) > 0:
        # Your format: result is a list with dictionaries
        for page_result in result:
            if isinstance(page_result, dict):
                print(f"   Processing dictionary result...")
                
                # Method 1: Check for 'rec_texts' key (your format)
                if 'rec_texts' in page_result:
                    texts = page_result['rec_texts']
                    scores = page_result.get('rec_scores', [1.0] * len(texts))
                    
                    print(f"   ✅ Found rec_texts with {len(texts)} items")
                    
                    for text, score in zip(texts, scores):
                        if text and text.strip():
                            extracted_texts.append((text.strip(), score))
                            print(f"      '{text}' (confidence: {score:.3f})")
                
                # Method 2: Check for old format compatibility
                elif isinstance(page_result, list):
                    print(f"   Processing list format...")
                    for detection in page_result:
                        if detection and len(detection) >= 2:
                            text_info = detection[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text = text_info[0]
                                confidence = text_info[1] if len(text_info) >= 2 else 1.0
                                
                                if text and text.strip():
                                    extracted_texts.append((text.strip(), confidence))
            
            # Handle old format (list of lists)
            elif isinstance(page_result, list):
                print(f"   Processing old list format...")
                for detection in page_result:
                    if detection and len(detection) >= 2:
                        text_info = detection[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) >= 2 else 1.0
                            
                            if text and text.strip():
                                extracted_texts.append((text.strip(), confidence))
    
    return extracted_texts

def convert_pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images for OCR processing
    """
    print(f"📄 Converting PDF to images...")
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"   📖 PDF opened: {len(doc)} pages")
        
        image_paths = []
        
        for page_num in range(len(doc)):
            print(f"   🔄 Converting page {page_num + 1}...")
            
            # Get page
            page = doc[page_num]
            
            # Convert to image with specified DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.tobytes("png")
            
            # Save as temporary image
            temp_image_path = f"temp_page_{page_num + 1}.png"
            with open(temp_image_path, "wb") as f:
                f.write(img_data)
            
            image_paths.append(temp_image_path)
            print(f"   ✅ Page {page_num + 1} saved as {temp_image_path}")
        
        doc.close()
        print(f"✅ PDF conversion complete: {len(image_paths)} images created")
        return image_paths
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return []

def process_pdf_with_ocr(pdf_path):
    """
    Process PDF file with OCR - extracts text from all pages
    """
    print("="*70)
    print("PADDLEOCR PDF TEXT EXTRACTION")
    print("="*70)
    
    # Check file
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []
    
    print(f"📄 Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, dpi=300)
    
    if not image_paths:
        print("❌ Failed to convert PDF to images")
        return []
    
    # Initialize OCR once for all pages
    print("🔧 Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='en')
    print("✅ PaddleOCR ready")
    
    all_extracted_texts = []
    page_results = {}
    
    try:
        # Process each page
        for page_num, image_path in enumerate(image_paths, 1):
            print(f"\n📄 Processing Page {page_num}...")
            
            # Load and check image
            img = cv2.imread(image_path)
            if img is None:
                print(f"   ❌ Cannot load image: {image_path}")
                continue
            
            print(f"   ✅ Image loaded: {img.shape}")
            
            # Resize if too large
            height, width = img.shape[:2]
            max_dimension = 1200
            
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized_img = cv2.resize(img, (new_width, new_height))
                resized_path = f"resized_page_{page_num}.jpg"
                cv2.imwrite(resized_path, resized_img)
                
                print(f"   📐 Resized from {width}x{height} to {new_width}x{new_height}")
                image_to_process = resized_path
            else:
                image_to_process = image_path
                print(f"   📐 Using original size: {width}x{height}")
            
            # Process with OCR
            print(f"   ⏳ Running OCR on page {page_num}...")
            start_time = time.time()
            
            try:
                result = ocr.ocr(image_to_process)
                process_time = time.time() - start_time
                print(f"   ✅ Page {page_num} processed in {process_time:.2f} seconds")
                
                # Extract text using our working method
                page_texts = extract_text_from_result(result)
                
                if page_texts:
                    print(f"   🎉 Page {page_num}: {len(page_texts)} text segments")
                    page_results[page_num] = page_texts
                    all_extracted_texts.extend(page_texts)
                else:
                    print(f"   ⚠️ Page {page_num}: No text extracted")
                    page_results[page_num] = []
                
            except Exception as e:
                print(f"   ❌ Error processing page {page_num}: {e}")
                page_results[page_num] = []
            
            # Cleanup resized image if created
            if image_to_process != image_path and os.path.exists(image_to_process):
                os.remove(image_to_process)
        
        # Display and save results
        if all_extracted_texts:
            print(f"\n🎉 SUCCESS! Extracted {len(all_extracted_texts)} total text segments from {len(image_paths)} pages")
            
            # Display summary per page
            print(f"\n📊 PAGE SUMMARY:")
            for page_num, texts in page_results.items():
                print(f"   Page {page_num}: {len(texts)} segments")
            
            # Display first few results
            print(f"\n📄 FIRST 10 TEXT SEGMENTS:")
            print("-" * 50)
            for i, (text, confidence) in enumerate(all_extracted_texts[:10], 1):
                print(f"{i:2d}. {text} (conf: {confidence:.3f})")
            
            if len(all_extracted_texts) > 10:
                print(f"   ... and {len(all_extracted_texts) - 10} more segments")
            
            # Save to file
            output_file = "pdf_extraction_results.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== PADDLEOCR PDF EXTRACTION RESULTS ===\n")
                f.write(f"PDF: {pdf_path}\n")
                f.write(f"Total pages: {len(image_paths)}\n")
                f.write(f"Total segments: {len(all_extracted_texts)}\n")
                f.write("="*50 + "\n\n")
                
                # Write page by page
                for page_num, texts in page_results.items():
                    f.write(f"\n--- PAGE {page_num} ---\n")
                    for text, confidence in texts:
                        f.write(f"{text} (confidence: {confidence:.3f})\n")
                
                # Write clean text (all pages combined)
                f.write("\n" + "="*50 + "\n")
                f.write("CLEAN TEXT (ALL PAGES COMBINED):\n")
                f.write("="*50 + "\n")
                for text, _ in all_extracted_texts:
                    f.write(f"{text}\n")
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Summary statistics
            total_chars = sum(len(text) for text, _ in all_extracted_texts)
            avg_confidence = sum(conf for _, conf in all_extracted_texts) / len(all_extracted_texts)
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"   • Total pages processed: {len(image_paths)}")
            print(f"   • Total text segments: {len(all_extracted_texts)}")
            print(f"   • Total characters: {total_chars}")
            print(f"   • Average confidence: {avg_confidence:.3f}")
            
            return [text for text, _ in all_extracted_texts]
        
        else:
            print("❌ No text extracted from any page")
            return []
    
    finally:
        # Cleanup all temporary image files
        print(f"\n🗑️ Cleaning up temporary files...")
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"   Deleted: {image_path}")
        
        print("✅ Cleanup complete")
def process_resume_with_correct_format(image_path):
    """
    Process single image with correct format handling (kept for backwards compatibility)
    """
    print("="*70)
    print("CORRECTED PADDLEOCR RESUME EXTRACTION")
    print("="*70)
    
    # Check file
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return []
    
    # Load and check image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return []
    
    print(f"✅ Image loaded: {img.shape}")
    
    # Resize if too large (your image is 1650x1275 which might be slow)
    height, width = img.shape[:2]
    max_dimension = 1200  # Reasonable size for OCR
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_img = cv2.resize(img, (new_width, new_height))
        resized_path = "resized_resume_for_ocr.jpg"
        cv2.imwrite(resized_path, resized_img)
        
        print(f"📐 Resized from {width}x{height} to {new_width}x{new_height}")
        image_to_process = resized_path
    else:
        image_to_process = image_path
        print(f"📐 Using original size: {width}x{height}")
    
    # Initialize OCR
    print("🔧 Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='en')
    print("✅ PaddleOCR ready")
    
    # Process image
    print("⏳ Processing image...")
    start_time = time.time()
    
    try:
        result = ocr.ocr(image_to_process)
        process_time = time.time() - start_time
        print(f"✅ Processing completed in {process_time:.2f} seconds")
        
        # Extract text using correct method
        extracted_texts = extract_text_from_result(result)
        
        if extracted_texts:
            print(f"\n🎉 SUCCESS! Extracted {len(extracted_texts)} text segments")
            
            # Display results
            print(f"\n📄 EXTRACTED TEXT:")
            print("-" * 50)
            for i, (text, confidence) in enumerate(extracted_texts, 1):
                print(f"{i:2d}. {text} (conf: {confidence:.3f})")
            
            # Save to file
            output_file = "corrected_resume_extraction.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== CORRECTED PADDLEOCR EXTRACTION ===\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Processing time: {process_time:.2f} seconds\n")
                f.write(f"Total segments: {len(extracted_texts)}\n")
                f.write("="*50 + "\n\n")
                
                for text, confidence in extracted_texts:
                    f.write(f"{text} (confidence: {confidence:.3f})\n")
                
                # Also save just the clean text
                f.write("\n" + "="*50 + "\n")
                f.write("CLEAN TEXT (no confidence scores):\n")
                f.write("="*50 + "\n")
                for text, _ in extracted_texts:
                    f.write(f"{text}\n")
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Show summary
            total_chars = sum(len(text) for text, _ in extracted_texts)
            avg_confidence = sum(conf for _, conf in extracted_texts) / len(extracted_texts)
            
            print(f"\n📊 SUMMARY:")
            print(f"   • Total text segments: {len(extracted_texts)}")
            print(f"   • Total characters: {total_chars}")
            print(f"   • Average confidence: {avg_confidence:.3f}")
            print(f"   • Processing time: {process_time:.2f} seconds")
            
            return [text for text, _ in extracted_texts]
        
        else:
            print("❌ No text extracted")
            print("🔍 DEBUG: Printing raw result structure...")
            print(f"Result: {result}")
            return []
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        # Cleanup
        if image_to_process != image_path and os.path.exists(image_to_process):
            os.remove(image_to_process)
            print(f"🗑️ Cleaned up temporary file: {image_to_process}")

if __name__ == "__main__":
    # Choose what to process
    pdf_path = "pdfs/A.pdf"  # Your PDF file
    image_path = "pdfs/Ali_Lazraq.jpg"  # Your image file
    
    print("🎯 PaddleOCR Text Extraction Tool")
    print("="*50)
    
    # Check what files exist
    pdf_exists = os.path.exists(pdf_path)
    image_exists = os.path.exists(image_path)
    
    print(f"📄 PDF file ({pdf_path}): {'✅ Found' if pdf_exists else '❌ Not found'}")
    print(f"🖼️ Image file ({image_path}): {'✅ Found' if image_exists else '❌ Not found'}")
    
    if pdf_exists:
        print(f"\n🚀 Processing PDF: {pdf_path}")
        result = process_pdf_with_ocr(pdf_path)
        
        if result:
            print(f"\n✅ PDF PROCESSING COMPLETE!")
            print(f"📊 Total text segments extracted: {len(result)}")
            print("\nFirst 5 segments:")
            for i, text in enumerate(result[:5], 1):
                print(f"   {i}. {text}")
        else:
            print("\n❌ PDF processing failed")
    
    elif image_exists:
        print(f"\n🚀 Processing Image: {image_path}")
        result = process_resume_with_correct_format(image_path)
        
        if result:
            print(f"\n✅ IMAGE PROCESSING COMPLETE!")
            print(f"📊 Total text segments extracted: {len(result)}")
            print("\nFirst 5 segments:")
            for i, text in enumerate(result[:5], 1):
                print(f"   {i}. {text}")
        else:
            print("\n❌ Image processing failed")
    
    else:
        print(f"\n❌ No files found to process!")
        print("Please ensure one of these files exists:")
        print(f"   • {pdf_path}")
        print(f"   • {image_path}")
    
    print("="*70)