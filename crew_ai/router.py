from openai import OpenAI
import base64


class GameVisionRouter:
    def __init__(self, routes: Dict[str, Dict[str, str]]):
        self.routes = routes

    def route(self, pred_label: str) -> Dict[str, str]:
        if pred_label not in self.routes:
            raise ValueError(f"No route configured for label: {pred_label}")
        return self.routes[pred_label]
    
    def image_to_data_url(self, image_path):
        suffix = image_path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def route_from_label(self, pred_label):
        if pred_label not in self.routes:
            raise ValueError(f"No route configured for label: {pred_label}")
        return self.routes[pred_label]

    def call_routed_llamacpp(self, image_path, pred_label, user_prompt):
        route = self.route_from_label(pred_label)
        client = OpenAI(
            base_url=route["base_url"].rstrip("/"),
            api_key="no-key-required",
        )

        image_url = self.image_to_data_url(image_path)

        resp = client.chat.completions.create(
            model=route["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.4,
        )

        return {
            "pred_label": pred_label,
            "routed_base_url": route["base_url"],
            "routed_model": route["model"],
            "response_text": resp.choices[0].message.content,
        }
    