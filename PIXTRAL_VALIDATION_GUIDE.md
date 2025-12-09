# 🔍 Pixtral MRI Image Validation Guide

## Overview

This implementation adds **two-stage AI validation** to prevent incorrect predictions on non-MRI images:

1. **Stage 1: Image Validation** - Pixtral AI analyzes if the image is a brain MRI
2. **Stage 2: Alzheimer's Prediction** - Your CNN model makes the diagnosis (only if Stage 1 passes)

---

## ✅ What's Implemented

### 1. Streamlit App (`app.py`)

**Automatic validation** has been added to the Diagnosis section:

```python
# When user uploads an image:
1. Image displayed
2. Pixtral validates it's a brain MRI
3. If INVALID → Show error, stop processing
4. If VALID → Proceed to CNN prediction
```

**User Experience:**
- ❌ **Invalid images** show clear error messages explaining why
- ✅ **Valid images** show success message before prediction
- 🔄 **API errors** show warnings but allow continuation

---

### 2. React/TypeScript Integration (`pixtral_validation.ts`)

A reusable TypeScript module for your React chat application.

**How to integrate into your `useChat` hook:**

```typescript
// 1. Import the validation function
import { validateMRIImage } from './pixtral_validation';

// 2. Add validation before sending to API
const handleSend = async () => {
  if (!input.trim() && !imagePreview) return;
  if (streaming) return;

  // VALIDATE IMAGE if present
  if (imagePreview) {
    setStreaming(true);
    const validation = await validateMRIImage(imagePreview);
    setStreaming(false);

    if (!validation.isValid) {
      // Show error in user's language
      const errorMsg = language === "ru"
        ? `❌ Недопустимое изображение: ${validation.reason}\n\nПожалуйста, загрузите МРТ-скан мозга.`
        : language === "kz"
        ? `❌ Жарамсыз кескін: ${validation.reason}\n\nМи МРТ сканерін жүктеңіз.`
        : `❌ Invalid image: ${validation.reason}\n\nPlease upload a brain MRI scan.`;

      alert(errorMsg);
      return; // STOP - don't proceed
    }

    // Optional: Show success message
    console.log("✅ Image validated:", validation.reason);
  }

  // Continue with normal message flow...
  const newMessage: Message = { /* ... */ };
  setMessages((prev) => [...prev, newMessage]);
  await callPixtralAPI(payload);
};
```

---

## 🧪 Testing the Validation

### Test Cases

Test with these image types to verify validation works:

#### ✅ Should PASS (Valid Brain MRI):
- Brain MRI scans (axial view)
- Brain MRI scans (sagittal view)
- Brain MRI scans (coronal view)
- Grayscale or colored medical brain imaging

#### ❌ Should FAIL (Invalid Images):
- Photos of people, cats, dogs, landscapes
- CT scans (different imaging modality)
- X-rays
- Knee/chest/abdomen MRI (different body part)
- Random screenshots
- Drawings or illustrations
- Completely blurred images

### Running Tests

**For Streamlit App:**
```bash
cd /home/corettaxkutcher/BrainSight-AI
streamlit run app.py
```

1. Navigate to "Diagnosis" tab
2. Upload a non-MRI image (e.g., a photo of a cat)
3. Should see: ❌ Error message stopping prediction
4. Upload a valid brain MRI
5. Should see: ✅ Validation success → Prediction results

**For React App:**
After integrating the validation:
1. Try uploading a regular photo → Should be rejected
2. Try uploading a brain MRI → Should proceed to prediction

---

## 🔧 Configuration

### API Key Management

**Current setup:** Hardcoded in files (for development)

**Production recommendation:** Use environment variables

```python
# app.py - Change to:
PIXTRAL_API_KEY = os.getenv("PIXTRAL_API_KEY", "your-fallback-key")
```

```typescript
// pixtral_validation.ts - Change to:
const PIXTRAL_API_KEY = process.env.REACT_APP_PIXTRAL_API_KEY || "your-fallback-key";
```

### Validation Strictness

You can adjust how strict the validation is:

**In `app.py` and `pixtral_validation.ts`, modify the validation criteria:**

```python
# More lenient - accept CT scans too
Criteria for a valid brain scan:
1. Must be a medical imaging scan
2. Must show brain structures
3. Can be MRI or CT scan  # <-- Changed

# More strict - require specific MRI types
Criteria for a valid brain MRI:
1. Must be T1 or T2 weighted MRI
2. Must show clear brain structures
3. Must be axial view only  # <-- More restrictive
```

---

## 🚀 Benefits

### Before (No Validation):
- ❌ Users upload cat photos → Wrong predictions
- ❌ Users upload chest X-rays → Nonsense results
- ❌ Confusion and loss of trust

### After (With Pixtral Validation):
- ✅ Invalid images rejected immediately
- ✅ Clear error messages guide users
- ✅ Only brain MRIs reach the CNN model
- ✅ Improved accuracy and user experience

---

## 📊 How Pixtral Validation Works

```
┌─────────────────────┐
│  User uploads image │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│   Pixtral AI analyzes image     │
│                                  │
│  - Is it medical imaging?        │
│  - Shows brain structures?       │
│  - MRI vs CT/X-ray?             │
│  - Proper orientation?           │
└──────────┬──────────────────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  INVALID      VALID
     │           │
     ▼           ▼
  ❌ Show     ✅ Continue
   Error      to CNN
              Prediction
```

---

## 🛠️ Troubleshooting

### "Validation service error"
- Check internet connection
- Verify Pixtral API key is correct
- Check API rate limits

### "Validation skipped due to error"
- Temporary API issue
- Prediction continues with warning (fail-safe behavior)

### Valid MRI rejected as invalid
- Image quality might be too low
- Try different MRI scan
- Check if image is corrupted

### Invalid image accepted as valid
- Edge case - report for improvement
- Validation confidence shown (HIGH/MEDIUM/LOW)

---

## 📝 API Response Examples

### Valid Brain MRI:
```
VALID: YES
CONFIDENCE: HIGH
REASON: This is a clear axial view brain MRI scan showing cerebral structures, ventricles, and gray/white matter differentiation.
```

### Invalid Image (Cat Photo):
```
VALID: NO
CONFIDENCE: HIGH
REASON: This is a photograph of a cat, not a medical imaging scan. No brain structures or MRI characteristics are present.
```

### Invalid Image (Chest X-ray):
```
VALID: NO
CONFIDENCE: HIGH
REASON: This appears to be a chest X-ray, not a brain MRI scan. Wrong imaging modality and anatomical region.
```

---

## 🎯 Next Steps

1. **Test thoroughly** with various image types
2. **Integrate into React** app using `pixtral_validation.ts`
3. **Monitor validation** success rates in production
4. **Collect feedback** from users on validation accuracy
5. **Fine-tune prompts** if needed for edge cases

---

## 📚 Additional Resources

- [Mistral Pixtral Documentation](https://docs.mistral.ai/)
- [BrainSight AI GitHub](https://github.com/your-repo)
- Medical Image Database for testing: [Radiopaedia](https://radiopaedia.org/)

---

## ⚠️ Important Notes

- Validation is **defensive**, not perfect
- Always include medical disclaimers
- This assists professionals, doesn't replace them
- Keep API keys secure in production
- Monitor API usage and costs

---

**Built with:**
- 🤖 Mistral Pixtral 12B (Vision AI)
- 🧠 PyTorch TinyVGG16 (Alzheimer's Classification)
- 🎨 Streamlit + React (Frontend)
- 💎 Google Gemini (Medical Assistant)
