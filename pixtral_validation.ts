/**
 * Pixtral Image Validation Module
 *
 * This module provides MRI image validation using Mistral's Pixtral vision AI
 * to ensure uploaded images are valid brain MRI scans before running predictions.
 *
 * Usage in React/TypeScript:
 * import { validateMRIImage } from './pixtral_validation';
 *
 * const result = await validateMRIImage(base64Image);
 * if (!result.isValid) {
 *   alert(result.reason);
 *   return;
 * }
 */

const PIXTRAL_API_KEY = "QqkMxELY0YVGkCx17Vya04Sq9nGvCahu";
const PIXTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions";

export interface ValidationResult {
  isValid: boolean;
  reason: string;
  confidence: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";
  rawResponse?: string;
}

/**
 * Validates if an image is a brain MRI scan using Pixtral vision AI
 *
 * @param imageBase64 - Base64 encoded image string (with or without data URL prefix)
 * @returns Promise<ValidationResult> - Validation result with detailed feedback
 */
export async function validateMRIImage(imageBase64: string): Promise<ValidationResult> {
  const validationPrompt = `Analyze this image carefully and determine if it is a brain MRI (Magnetic Resonance Imaging) scan.

You must respond in this EXACT format:
VALID: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [Brief explanation]

Criteria for a valid brain MRI:
1. Must be a medical imaging scan (grayscale or colored medical imaging)
2. Must show brain structures (cerebral cortex, ventricles, white/gray matter)
3. Must be an MRI scan (not CT, X-ray, ultrasound, or other imaging types)
4. Should be a proper axial, sagittal, or coronal brain view
5. Not a photograph, drawing, or non-medical image

Examples of INVALID images:
- Photos of people, animals, objects, landscapes
- Other body part scans (knee, chest, abdomen MRI)
- CT scans, X-rays, ultrasounds
- Low quality or completely blurred images
- Drawings or illustrations`;

  const payload = {
    model: "pixtral-12b-2409",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: validationPrompt },
          { type: "image_url", image_url: { url: imageBase64 } }
        ]
      }
    ],
    temperature: 0.3,
    top_p: 1,
    stream: false
  };

  try {
    const response = await fetch(PIXTRAL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${PIXTRAL_API_KEY}`
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Pixtral API Error:", errorText);
      return {
        isValid: false,
        reason: `Validation service error (Status ${response.status})`,
        confidence: "LOW",
        rawResponse: errorText
      };
    }

    const data = await response.json();
    const resultText = data?.choices?.[0]?.message?.content || "";

    // Parse the response
    const isValid = resultText.toUpperCase().includes("VALID: YES");

    // Extract confidence and reason
    const lines = resultText.trim().split('\n');
    let confidence: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN" = "UNKNOWN";
    let reason = "No reason provided";

    for (const line of lines) {
      const upperLine = line.toUpperCase();
      if (upperLine.includes("CONFIDENCE:")) {
        const confText = line.split(':', 2)[1]?.trim().toUpperCase();
        if (confText === "HIGH" || confText === "MEDIUM" || confText === "LOW") {
          confidence = confText as "HIGH" | "MEDIUM" | "LOW";
        }
      } else if (upperLine.includes("REASON:")) {
        reason = line.split(':', 2)[1]?.trim() || reason;
      }
    }

    return {
      isValid,
      reason,
      confidence,
      rawResponse: resultText
    };

  } catch (error: any) {
    console.error("Validation error:", error);
    return {
      isValid: false,
      reason: `Validation failed: ${error.message}`,
      confidence: "LOW"
    };
  }
}

/**
 * Example usage in your useChat hook:
 *
 * ```typescript
 * const handleSend = async () => {
 *   if (!input.trim() && !imagePreview) return;
 *   if (streaming) return;
 *
 *   // VALIDATE IMAGE FIRST if image is present
 *   if (imagePreview) {
 *     setStreaming(true); // Show loading state
 *     const validation = await validateMRIImage(imagePreview);
 *     setStreaming(false);
 *
 *     if (!validation.isValid) {
 *       const errorMsg = language === "ru"
 *         ? `Недопустимое изображение: ${validation.reason}`
 *         : language === "kz"
 *         ? `Жарамсыз кескін: ${validation.reason}`
 *         : `Invalid image: ${validation.reason}`;
 *
 *       alert(errorMsg);
 *       return; // Stop - don't send to model
 *     }
 *   }
 *
 *   // Continue with normal flow...
 *   const newMessage: Message = {
 *     sender: "user",
 *     text: input,
 *     imageUrl: imagePreview || undefined,
 *     timestamp: new Date(),
 *     id: generateMessageId()
 *   };
 *
 *   setMessages((prev) => [...prev, newMessage]);
 *   // ... rest of your code
 * };
 * ```
 */
