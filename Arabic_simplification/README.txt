الصافي | Al-Safy

A web app that takes Arabic legal documents — either as an image or plain text — and rewrites them in simpler Arabic that's easier to understand.

Built as a university capstone project by a team of three.


What it does

You upload a photo of a legal document (or paste the text directly), and the app returns a simplified version of it. Under the hood, a fine-tuned Arabic language model handles the simplification, and OCR handles pulling text out of images.

Stack


Model: AraBART, fine-tuned on Jordanian legal text
Backend: FastAPI
Frontend: React
OCR: Mistral OCR