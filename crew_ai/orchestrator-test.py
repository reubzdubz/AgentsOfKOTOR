from router import GameVisionRouter
import requests

CLASSIFIER_ENDPOINT = "http://localhost:8080"
ROUTES =  {
    "combat": {"base_url": "http://localhost:5000", "model": "combat-specialist"},
    "narrative": {"base_url": "http://localhost:5001", "model": "narrative-specialist"},
    "leveling": {"base_url": "http://localhost:5002", "model": "leveling-specialist"},
}
TEST_SCREENSHOT_PATH = "vision_system/datasets/kotor_ui_samples/combat/sample_0202.jpg"

router = GameVisionRouter(ROUTES)

game_state = requests.post(
    f"{CLASSIFIER_ENDPOINT}/analyze", files={"file": open(TEST_SCREENSHOT_PATH, "rb")}, 
    data={"prompt": "Classify the following KOTOR user interface screenshot into exactly one of these categories: combat, narrative, or leveling. Only answer with one word: combat, narrative, or leveling."}
)
pred_label = game_state.text.strip()

for token in router.call_routed_llamacpp(TEST_SCREENSHOT_PATH, pred_label, "Describe the current game state based on the screenshot, and suggest the best next action for the player to take."):
    print(token, end="", flush=True)
