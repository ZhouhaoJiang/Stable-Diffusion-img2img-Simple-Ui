import os
import time
import gradio as gr
import gradio.helpers

from services.img2img_mask import *
from services.deep_danbooru import *

examples = [
    os.path.join(os.path.dirname(__file__), "examples/test01.png"),
    os.path.join(os.path.dirname(__file__), "examples/test02.png"),
    os.path.join(os.path.dirname(__file__), "examples/test03.png"),
    os.path.join(os.path.dirname(__file__), "examples/test04.png"),
]

with gr.Blocks(title="Repaint") as demo:
    gr.Markdown(
        """
            <center><h1>Stable Diffusion局部重绘 demo演示</hq><center>
        """
    )

    with gr.Row():
        prompt = gr.Textbox(label="正向提示词", lines=2)
        negative_prompt = gr.Textbox(label="反向提示词", lines=2)
        with gr.Column(scale=1):
            deep_button = gr.Button(value="反向推理词(须先上传图片)",variant="primary")
            generate_button = gr.Button(value="生成图片",variant="primary")

    with gr.Row():
        steps = gr.Slider(label="迭代次数(图片质量)", minimum=10, maximum=50, step=1, value=25)
        batch_size = gr.Slider(label="图片生成数量", minimum=1, maximum=4, step=1, value=1)

    with gr.Row():
        img_and_mask = gr.Image(label="输入图片", interactive=True, tool='sketch', type="pil",brush_radius=80,
                                width=512, height=512, show_label=True)
        # output = gr.Image(label="输出图片", interactive=False, type="pil", width=512, height=512)
        output_images = gr.Gallery(label="输出图片")

    with gr.Row():
        gr.Examples(examples, inputs=img_and_mask)

    generate_button.click(get_pic, inputs=[img_and_mask, prompt, negative_prompt, steps, batch_size], outputs=output_images)
    deep_button.click(deepdanbooru, inputs=[img_and_mask], outputs=prompt)


# 在0.0.0.0:7788端口启动demo
if __name__ == '__main__':
    demo.launch(inline=False, share=False, debug=True, server_name="0.0.0.0", server_port=7788)
