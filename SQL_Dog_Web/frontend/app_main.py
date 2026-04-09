def main():
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css
    )
