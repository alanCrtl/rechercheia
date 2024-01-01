"""
task-2 implementation

TODO: implement consistuency parser
NOTE: 	use this https://demo.allennlp.org/constituency-parsing
		and this https://demo.allennlp.org/dependency-parsing

======
Example data from training set:
sentenceID	sentence	antecedent	consequent	antecedent_startid	antecedent_endid	consequent_startid	consequent_endid
200000	I don't think any of us---even economic gurus like Paul Krugman---really, truly understand just how bad it could've gotten "on Main Street" if the stimulus bill had become hamstrung by a filibuster threat or recalcitrant conservadems, the way so much of our legislation has since.	if the stimulus bill had become hamstrung by a filibuster threat or recalcitrant conservadems	I don't think any of us---even economic gurus like Paul Krugman---really, truly understand just how bad it could've gotten "on Main Street"	140	232	0	138

======
CHATGPT after giving notebook in raw input:
(https://chat.openai.com/c/de3368c5-c17a-4266-9c52-48cfd6fa3a81)

It looks like you've shared a code snippet written in Python. The code appears to be a part of a script or notebook related to natural language processing tasks. The functions defined in the code seem to be involved in processing and analyzing text, particularly for extracting modal verbs and antecedents.

Here's a brief overview of the functions:

defragment(fragments, text, original, removed):

Takes a list of fragments, the original text, and removed characters as input.
Sorts fragments by length.
Tries to find the fragments in the text, and if found, overlays them on the original text.
Handles cases where fragments overlap or have no overlap.
Updates the labels for the result.
overlay(starting, fragment, original, removed):

Takes a starting position, a fragment dictionary, the original text, and removed characters as input.
Offsets the fragment's text based on the starting position.
Adjusts the start and end positions considering removed characters.
Returns the updated fragment.
offset(text, starting, original):

Takes a text, starting position, and original text as input.
Finds the offset positions of the text in the original text.
Handles cases where characters are removed.
record(original):

Takes the original text as input.
Records the indices and characters of the original text in a list.
contract(text, base):

Takes a text and a base (recorded indices and characters) as input.
Identifies and removes contracted phrases.
separate_from(md, pa):

Takes two strings as input (md and pa).
Separates the substring from md occurring before pa.
separate_right(md, pa):

Takes two strings as input (md and pa).
Separates the substring from md occurring after pa.
separate_left(md, pa):

Takes two strings as input (md and pa).
Separates the substring from md occurring before pa.
separate(modal, past):

Takes lists of modal and past strings as input.
Separates strings in modal and past lists based on their overlapping substrings.
deoverlap(past):

Takes a list of past strings as input.
Removes redundant overlapping substrings.
focus(spans):

Takes a list of spans as input.
Focuses on specific spans based on their positions.
chunk(original):

Takes the original text as input.
Extracts verbs, modals, pasts, and other linguistic components from the text.
Handles overlap and separation of linguistic components.
It seems like these functions are used to process and analyze text, particularly for extracting information related to modals and antecedents. If you have specific questions or tasks you'd like assistance with, feel free to ask!
"""

