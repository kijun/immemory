import re
from typing import List, Optional

class MyBeat:
    def __init__(self):
        self.actions = []     # type: List[str]
        self.dialogues = []   # type: List[(character, line)]

class MySubscene:
    def __init__(self):
        self.beats = []       # type: List[MyBeat]

class MyScene:
    def __init__(self, heading: str, scene_number: Optional[str] = None):
        self.heading = heading
        self.scene_number = scene_number
        self.subscenes = []   # type: List[MySubscene]

class MyScript:
    def __init__(self):
        self.scenes = []      # type: List[MyScene]

class ScriptBuilder:
    def __init__(self):
        self.script = MyScript()

        self.current_scene = None    # type: Optional[MyScene]
        self.current_subscene = None # type: Optional[MySubscene]
        self.current_beat = None     # type: Optional[MyBeat]

        # A naive check for transitions that trigger new subscenes.
        self.transition_regex = re.compile(r'^(CUT TO:|LATER|FLASHBACK|END FLASHBACK|FADE TO BLACK|SMASH CUT TO:).*', re.IGNORECASE)

    def start_new_scene(self, heading: str, scene_number: Optional[str]):
        """Create a new scene and subscene+beat chain."""
        self.current_scene = MyScene(heading, scene_number)
        self.script.scenes.append(self.current_scene)

        self.current_subscene = MySubscene()
        self.current_scene.subscenes.append(self.current_subscene)

        self.current_beat = MyBeat()
        self.current_subscene.beats.append(self.current_beat)

    def start_new_subscene(self):
        """Create a new subscene and new beat."""
        if not self.current_scene:
            # If we haven't started a scene yet, create a dummy one
            self.start_new_scene("UNTITLED SCENE", None)
            return

        self.current_subscene = MySubscene()
        self.current_scene.subscenes.append(self.current_subscene)

        self.current_beat = MyBeat()
        self.current_subscene.beats.append(self.current_beat)

    def ensure_beat_exists(self):
        """Make sure we have a current beat to store actions/dialogues in."""
        if not self.current_beat:
            # If no subscene or beat, create them now
            if not self.current_subscene:
                self.start_new_subscene()
            else:
                self.current_beat = MyBeat()
                self.current_subscene.beats.append(self.current_beat)

    # ---- Callback Methods ----

    def on_scene_heading(self, text: str, scene_number: str):
        # Start a new scene (and automatically new subscene+beat)
        self.start_new_scene(text, scene_number if scene_number else None)

    def on_action(self, text: str):
        self.ensure_beat_exists()
        self.current_beat.actions.append(text)

    def on_dialogue(self, character: str, extension: str, parenthetical: str, line: str, is_dual_dialogue: bool):
        self.ensure_beat_exists()

        # Combine extension and parenthetical into a single "character" field if you like
        # or store them separately. We'll do a naive approach:
        full_character = character
        if extension:
            full_character += f" ({extension})"
        if parenthetical:
            full_character += f" [{parenthetical}]"

        self.current_beat.dialogues.append((full_character, line))

    def on_transition(self, text: str):
        """If text matches our naive transition triggers, start a new subscene."""
        if self.transition_regex.match(text):
            self.start_new_subscene()

    # The rest are optional if you want to handle them:
    def on_lyrics(self, text: str):
        self.ensure_beat_exists()
        self.current_beat.actions.append(f"LYRICS: {text}")

    def on_section(self, text: str, level: int):
        # Could store as a special marker if you want
        pass

    def on_synopsis(self, text: str):
        pass

    def on_page_break(self):
        pass

    def on_title_page(self, entries):
        # e.g., store them or print them
        pass

from fountain_tools.callback_parser import CallbackParser, TitleEntry
from fountain_tools.parser import Parser as FountainParser
from fountain_tools.writer import Writer as FountainWriter
from fountain_tools.fountain import *

def parse_fountain_into_structure(fountain_text: str) -> MyScript:
    # Create parser and builder
    parser = CallbackParser()
    builder = ScriptBuilder()

    # Assign callbacks
    parser.onTitlePage = builder.on_title_page
    parser.onSceneHeading = builder.on_scene_heading
    parser.onAction = builder.on_action
    parser.onDialogue = builder.on_dialogue
    parser.onTransition = builder.on_transition
    parser.onLyrics = builder.on_lyrics
    parser.onSection = builder.on_section
    parser.onSynopsis = builder.on_synopsis
    parser.onPageBreak = builder.on_page_break

    # Parse the text
    parser.add_text(fountain_text)
    parser.finalize()

    # Return the final data structure
    return builder.script

from fountain_tools.parser import Parser as FountainParser
from fountain_tools.writer import Writer as FountainWriter
from fountain_tools.fountain import *

def print_script_structure(my_script: MyScript):
    """
    Print the script's structure:
      - Scenes
        - Subscenes
          - Beats
            - Actions
            - Dialogues
    """
    for s_i, scene in enumerate(my_script.scenes, start=1):
        print(f"SCENE {s_i}: {scene.heading} (Scene #: {scene.scene_number})")
        for sub_i, subscene in enumerate(scene.subscenes, start=1):
            print(f"  SUBSCENE {sub_i}")
            for b_i, beat in enumerate(subscene.beats, start=1):
                print(f"    BEAT {b_i}")
                if beat.actions:
                    print(f"      Actions ({len(beat.actions)})")
                    for a in beat.actions:
                        print(f"        - {a}")
                if beat.dialogues:
                    print(f"      Dialogues ({len(beat.dialogues)})")
                    for char, dlg in beat.dialogues:
                        print(f"        {char}: {dlg}")

def main():
    from_narration_file = "sample_screenplay.fountain"
    narration_script = ""

    if from_narration_file:
        print(f"Loading narration text from {from_narration_file}...")
        try:
            with open(from_narration_file, "r", encoding="utf-8") as f:
                narration_script = f.read()
        except Exception as e:
            print("Failed to load narration file:", e)
            narration_script = ""

    script = parse_fountain_into_structure(narration_script)
    print(script)
    print_script_structure(script)



if __name__ == "__main__":
    main()