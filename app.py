from pathlib import Path
import cv2 as cv
import numpy as np
import streamlit as st
from pipeline.interface import DiceInferencePipeline, draw_annotated
from pipeline.scoring import (
    CATEGORIES, Scorecard, all_category_scores, UPPER_CATEGORIES,
)

DICE_FACES = {1: "\u2680", 2: "\u2681", 3: "\u2682", 4: "\u2683", 5: "\u2684", 6: "\u2685"}
CATEGORY_LABELS = {
    "ones": "Ones", "twos": "Twos", "threes": "Threes", "fours": "Fours",
    "fives": "Fives", "sixes": "Sixes", "three_of_a_kind": "3 of a Kind",
    "four_of_a_kind": "4 of a Kind", "full_house": "Full House",
    "small_straight": "Small Straight", "large_straight": "Large Straight",
    "yahtzee": "YAHTZEE", "chance": "Chance",
}
LOW_CONFIDENCE_THRESHOLD = 0.60
MAX_ROLLS_PER_TURN = 3

st.set_page_config(page_title="RollCall", page_icon="\U0001F3B2", layout="wide")

@st.cache_resource
def load_pipeline(checkpoint_path: str):
    return DiceInferencePipeline(checkpoint_path)

def get_pipeline():
    checkpoint_path = st.session_state.get("checkpoint_path", "")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    try:
        return load_pipeline(checkpoint_path)
    except Exception as e:
        st.session_state["pipeline_error"] = str(e)
        return None

def init_state():
    defaults = {
        "checkpoint_path": "runs/custom/run_20/best_model.pt",
        "scorecard": Scorecard(),
        "dice": [None] * 5,
        "kept": [False] * 5,
        "confidences": [None] * 5,
        "rolls_used": 0,
        "last_annotated": None,
        "turn_complete_prompt": False,
        "turn_id": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

def start_new_turn():
    st.session_state["dice"] = [None] * 5
    st.session_state["kept"] = [False] * 5
    st.session_state["confidences"] = [None] * 5
    st.session_state["rolls_used"] = 0
    st.session_state["last_annotated"] = None
    st.session_state["turn_complete_prompt"] = False
    st.session_state["turn_id"] += 1

def open_slots():
    return [i for i, k in enumerate(st.session_state["kept"]) if not k]

def apply_detections_to_slots(detections, slot_indices):
    n = min(len(detections), len(slot_indices))
    for i in range(n):
        slot = slot_indices[i]
        st.session_state["dice"][slot] = detections[i].predicted_face
        st.session_state["confidences"][slot] = detections[i].confidence
        st.session_state.pop(f"manual_{slot}_{st.session_state['turn_id']}", None)
    return len(detections), len(slot_indices)

with st.sidebar:
    st.header("Setup")
    st.session_state["checkpoint_path"] = st.text_input(
        "Model checkpoint path", value=st.session_state["checkpoint_path"],
        help="Path to your trained DiceClassifier weights (.pt)",
    )
    pipeline = get_pipeline()
    if pipeline is None:
        st.warning(
            "No model loaded yet. Point this at a real checkpoint to run detection, or use manual entry below."
        )

    st.divider()
    st.header("Scorecard")
    sc: Scorecard = st.session_state["scorecard"]
    upper_rows = [c for c in CATEGORIES if c in UPPER_CATEGORIES]
    lower_rows = [c for c in CATEGORIES if c not in UPPER_CATEGORIES]

    for cat in upper_rows:
        val = sc.entries[cat]
        st.write(f"{CATEGORY_LABELS[cat]}: {'—' if val is None else val}")
    st.write(f"**Upper subtotal:** {sc.upper_subtotal()}")
    st.write(f"**Upper bonus:** {sc.upper_bonus()}")
    st.divider()
    for cat in lower_rows:
        val = sc.entries[cat]
        st.write(f"{CATEGORY_LABELS[cat]}: {'—' if val is None else val}")
    st.divider()
    st.subheader(f"Total: {sc.grand_total()}")

    if sc.is_complete():
        st.success("Game complete!")
        if st.button("Start new game"):
            st.session_state["scorecard"] = Scorecard()
            start_new_turn()
            st.rerun()

st.title("\U0001F3B2 RollCall")
st.caption("Photograph your dice..RollCall will read them for you!")

if st.session_state.get("pipeline_error"):
    st.error(f"Model failed to load: {st.session_state['pipeline_error']}")

rolls_used = st.session_state["rolls_used"]
slots_to_fill = open_slots() if rolls_used > 0 else list(range(5))
rolls_left = MAX_ROLLS_PER_TURN - rolls_used

st.subheader(
    f"Roll {rolls_used + 1} of {MAX_ROLLS_PER_TURN}"
    if rolls_used < MAX_ROLLS_PER_TURN else "Rolls used up"
)

if rolls_used < MAX_ROLLS_PER_TURN:
    n_expected = len(slots_to_fill)
    label = (
        f"Photograph all {n_expected} dice"
        if rolls_used == 0 else
        f"Photograph the {n_expected} die you're rerolling"
        if n_expected == 1 else
        f"Photograph the {n_expected} dice you're rerolling"
    )
    photo_file = st.file_uploader(
        label, type=["jpg", "jpeg"],
        key=f"upload_{rolls_used}_{st.session_state['turn_id']}",
    )

    if photo_file is not None:
        file_bytes = np.frombuffer(photo_file.read(), np.uint8)
        photo_bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        if pipeline is not None:
            detections, _ = pipeline.process_photo(
                photo_bgr, expected_dice_count=len(slots_to_fill), debug=False
            )
            n_detected, n_expected = apply_detections_to_slots(detections, slots_to_fill)
            st.session_state["last_annotated"] = cv.cvtColor(
                draw_annotated(photo_bgr, detections), cv.COLOR_BGR2RGB
            )
            if n_detected != n_expected:
                st.warning(
                    f"Expected {n_expected} dice but located {n_detected} in the photo. "
                    "Use the manual override below to correct any slot before rerolling or scoring."
                )
            st.session_state["rolls_used"] += 1
            st.rerun()
        else:
            st.info("No model loaded — enter values manually below for each die.")

if st.session_state["last_annotated"] is not None:
    with st.expander("What RollCall saw", expanded=False):
        st.image(st.session_state["last_annotated"], use_container_width=True)

st.divider()
st.subheader("Your dice")

cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        face = st.session_state["dice"][i]
        conf = st.session_state["confidences"][i]
        display = DICE_FACES.get(face, "\u2753") if face else "\u2753"
        st.markdown(f"<div style='font-size:72px;text-align:center'>{display}</div>", unsafe_allow_html=True)

        if conf is not None and conf < LOW_CONFIDENCE_THRESHOLD:
            st.caption(f"\u26A0\uFE0F low confidence ({conf:.0%})")
        elif conf is not None:
            st.caption(f"confidence: {conf:.0%}")

        manual_key = f"manual_{i}_{st.session_state['turn_id']}"
        manual = st.selectbox(
            "Correct value", options=["", 1, 2, 3, 4, 5, 6],
            index=0, key=manual_key, label_visibility="collapsed",
        )
        if manual != "":
            st.session_state["dice"][i] = int(manual)
            st.session_state["confidences"][i] = None
            face = int(manual)

        keep_disabled = face is None
        keep_key = f"keep_{i}_{st.session_state['turn_id']}"
        kept = st.checkbox(
            "Keep", value=st.session_state["kept"][i], key=keep_key,
            disabled=keep_disabled,
        )
        st.session_state["kept"][i] = kept

all_slots_filled = all(v is not None for v in st.session_state["dice"])
rolls_left_after = MAX_ROLLS_PER_TURN - st.session_state["rolls_used"]

st.divider()
action_cols = st.columns(2)

with action_cols[0]:
    if all_slots_filled and rolls_left_after > 0 and any(not k for k in st.session_state["kept"]):
        st.write(f"{rolls_left_after} reroll(s) remaining. Upload a new photo above for your non-kept dice.")
    elif rolls_left_after == 0:
        st.write("No rerolls left this turn — time to score.")

with action_cols[1]:
    if all_slots_filled:
        st.markdown("### Score this roll")
        dice_values = st.session_state["dice"]
        scores = all_category_scores(dice_values)
        sc: Scorecard = st.session_state["scorecard"]
        open_cats = sc.open_categories()

        chosen = st.radio(
            "Choose a category",
            options=open_cats,
            format_func=lambda c: f"{CATEGORY_LABELS[c]} — {scores[c]} pts",
        )
        if st.button("Confirm score", type="primary"):
            sc.record(chosen, dice_values)
            start_new_turn()
            st.rerun()