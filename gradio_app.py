import gradio as gr

# Function to perform inference

def inference(input_text):
    # Your inference logic here
    return f"Inference result for: {input_text}"

# Create a generator for yielding progress

async def infer_and_yield_progress(input_text):
    with gr.Progress() as progress:
        for i in range(1, 5):  # Simulate processing steps
            # Update progress
            progress(100 * i / 4, f"Step {i} out of 4")
            await asyncio.sleep(1)  # Simulate delay
    return f"Inference completed for: {input_text}"

# Create Gradio interface

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Inference Demo")

        with gr.TabbedInterface():
            with gr.Tab("Inference"):
                input_text = gr.Textbox(label="Input Text")
                output = gr.Textbox(label="Output")
                btn_infer = gr.Button("Run Inference")

                btn_infer.click(fn=infer_and_yield_progress, inputs=input_text, outputs=output)
            
    demo.queue(concurrency_count=1, max_size=20)
    demo.launch()

if __name__ == "__main__":
    main()