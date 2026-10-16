import gradio as gr

from chat import ChatComponent


def respond(message, history, state):
    if state is None:
        state = ChatComponent()

    message = (message or "").strip()
    if not message:
        yield history, state, ""
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    for partial in state.send_message(message):
        history[-1]["content"] = partial
        yield history, state, ""


def reset():
    return [], ChatComponent(), ""


with gr.Blocks(title="智能助手") as demo:
    state = gr.State(value=None)

    gr.Markdown(
        """
        # 🤖 智能助手
        一个基于 DeepSeek 的聊天机器人。
        """
    )

    chatbot = gr.Chatbot(
        label="对话",
        height=460,
        avatar_images=(None, "🤖"),
        placeholder="哼，还不快向本座提问！",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="输入消息，按回车发送 ...",
            label="你的消息",
            lines=2,
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

    gr.Examples(
        examples=["你好", "给我讲个笑话", "什么是AIGC？"],
        inputs=msg,
        label="试试这些",
    )

    msg.submit(respond, [msg, chatbot, state], [chatbot, state, msg])
    send_btn.click(respond, [msg, chatbot, state], [chatbot, state, msg])
    clear_btn.click(reset, outputs=[chatbot, state, msg])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
