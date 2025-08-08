from gradio_client import Client, handle_file
import gradio as gr
import re
import os


client = Client("yisol/IDM-VTON",
                hf_token=os.getenv("HF_TOKEN"))


def vton_generation(human_model_img: str, garment: str):
    """Use the IDM-VTON model to generate a new image of the person wearing a garment."""
    """
    Args:
        human_model_img: The human model that is modelling the garment.
        garment: The garment to wear.
    """

    output = client.predict(
        dict={"background": handle_file(human_model_img), "layers": [], "composite": None},
        garm_img=handle_file(garment),
        garment_des="",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    return output[0]


vton_mcp = gr.Interface(
    vton_generation,
    inputs=[
        gr.Image(type="filepath", label= "Human Model Image URL"),
        gr.Image(type="filepath", label="Garment Image URL or File")
    ],
    outputs=gr.Image(type="filepath", label="Generated Image")
)

if __name__ == "__main__":
    vton_mcp.launch(mcp_server=True)