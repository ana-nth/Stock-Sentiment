import gradio as gr
from my_utils import final_output, generate_audio

def analyze_and_get_audio(company):
    """Fetches sentiment analysis and generates an audio summary."""
    analysis = final_output(company)
    summary = analysis["Final Sentiment Analysis"]

    # Generate audio
    audio_path = generate_audio(summary, company)

    return summary, audio_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📰 News Sentiment Analysis")

    with gr.Row():
        company_input = gr.Textbox(label="Enter Company Name")
        analyze_button = gr.Button("Analyze Sentiment")

    sentiment_output = gr.Textbox(label="Sentiment Analysis Result")
    audio_output = gr.Audio()

    analyze_button.click(analyze_and_get_audio, inputs=[company_input], outputs=[sentiment_output, audio_output])

# Launch Gradio App
demo.launch()
