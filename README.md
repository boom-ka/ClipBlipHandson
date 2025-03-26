# 🖼️ CLIP & BLIP Hands-On Implementation

This project demonstrates how to use **CLIP (Contrastive Language-Image Pretraining)** and **BLIP (Bootstrapped Language-Image Pretraining)** for **image-text matching** and **visual question answering (VQA)**.  

---

## 📌 Overview

- **CLIP**: Matches images with relevant text descriptions.  
- **BLIP**: Answers questions based on an image.  
- **Stable Diffusion**: Generates an image from a text prompt.  

---

## 🚀 Installation

Run the following commands to install the required dependencies:




🖼️ Step 1: Generate an Image with Stable Diffusion
---------------------------------------------------

Use **Stable Diffusion** from Hugging Face to create an image from a text prompt.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`pythonCopyEditfrom diffusers import StableDiffusionPipeline  import torch    # Load the model  pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")    pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available    # Generate an image from a prompt  prompt = "A futuristic city with flying cars"    image = pipe(prompt).images[0]    image.save("generated_image.png")  # Save the generated image    image.show()  # Display the image`  

🎯 Step 2: Use CLIP to Match Text with the Generated Image
----------------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopyEditimport clip    import torch    from PIL import Image    # Load CLIP model  device = "cuda" if torch.cuda.is_available() else "cpu"  model, preprocess = clip.load("ViT-B/32", device=device)  # Load the generated image  image = preprocess(Image.open("generated_image.png")).unsqueeze(0).to(device)  # Define text descriptions  text_descriptions = ["A futuristic city", "A forest", "A sunset over the ocean"]  text_tokens = clip.tokenize(text_descriptions).to(device)  # Compute similarity  with torch.no_grad():      image_features = model.encode_image(image)      text_features = model.encode_text(text_tokens)      similarity = (image_features @ text_features.T).softmax(dim=-1)  # Print best matching text  best_match = text_descriptions[similarity.argmax()]  print(f"Best Matching Description: {best_match}")   `

❓ Step 3: Use BLIP for Visual Question Answering (VQA)
------------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopyEditfrom transformers import BlipProcessor, BlipForQuestionAnswering  from PIL import Image  import torch  # Load BLIP VQA model  device = "cuda" if torch.cuda.is_available() else "cpu"  processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")  model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)  # Load the generated image  image = Image.open("generated_image.png")  # Ask a question  question = "What is in the image?"  inputs = processor(images=image, text=question, return_tensors="pt").to(device)  # Generate an answer  answer_ids = model.generate(**inputs)  answer = processor.decode(answer_ids[0], skip_special_tokens=True)  print(f"BLIP Answer: {answer}")   `

🎯 Expected Output
------------------

After running the script, you should get:

1.  **An image** generated using **Stable Diffusion**.
    
2.  **CLIP** will find the **best-matching text description**.
    
3.  **BLIP** will **answer a question** about the image.
    

### Example Output:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   lessCopyEditBest Matching Description: A futuristic city  BLIP Answer: A city with flying cars   `

🔍 Explanation of Each Step
---------------------------

### 1️⃣ **Stable Diffusion**

*   Generates an image based on the given **text prompt**.
    
*   The image is saved as "generated\_image.png".
    

### 2️⃣ **CLIP**

*   Loads the **pretrained CLIP model**.
    
*   Takes an image and multiple text descriptions as input.
    
*   Computes similarity and finds the **best-matching text description** for the generated image.
    

### 3️⃣ **BLIP (Visual Question Answering)**

*   Loads the **pretrained BLIP model**.
    
*   Takes the image and a **question** as input.
    
*   **Generates an answer** based on the image content.
    

📌 Conclusion
-------------

*   **CLIP** helps match images to the most relevant text.
    
*   **BLIP VQA** answers questions based on an image.
    
*   **Stable Diffusion** generates images from text prompts.
    
*   These models together demonstrate **AI’s ability to understand and process images & text** without labeled training data.
    

🔗 References
-------------

*   [CLIP GitHub](https://github.com/openai/CLIP)
    
*   [BLIP GitHub](https://github.com/salesforce/BLIP)
    
*   Stable Diffusion on Hugging Face
