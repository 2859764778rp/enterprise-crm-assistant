from models import chat

class ChatComponent:
    def __init__(self):
        self.messages = [{"role": "system", "content":"你是一个智能助手。"}]

    def send_message(self, message):
        self.messages.append({"role": "user", "content": message})
        response = ""
        for delta in chat(self.messages):
            response += delta
            yield response
        self.messages.append({"role": "assistant", "content": response})

    def get_history(self):
        return self.messages[1:]

if __name__ == "__main__":
    chat_component = ChatComponent()
    while True:
        User_input = input("用户:")
        if User_input.lower() == "exit":
            break
        print("助手:", end="", flush=True)
        previous = ""
        for partial in chat_component.send_message(User_input):
            print(partial[len(previous):], end="", flush=True)
            previous = partial
        print()
