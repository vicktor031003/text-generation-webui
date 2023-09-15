import gradio as gr

from modules import shared, ui, utils, presets
from modules.github import clone_or_pull_repository
from modules.utils import gradio


def create_ui(default_preset):
    generate_params = presets.load_preset(default_preset)
    shared.gradio['textbox-notebook'] = gr.Textbox(value='', lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'])
    shared.gradio['prompt_menu-notebook'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
    shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
    shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Expand max_new_tokens to the available context length.')
    shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='Maximum number of tokens/second', info='To make text readable in real time.')
    shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')
    shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature')
    shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
    shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
    shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
    shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
    shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')
    shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
    shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
    shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
    shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
    shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
    shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
    shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length')
    shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
    shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.')
    shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams', info='For Beam Search, along with length_penalty and early_stopping.')
    shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
    shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')
    shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.')
    shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
    shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
    shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='For CFG. 1.5 is a good value.')
    shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='Negative prompt', lines=3, elem_classes=['add_scrollbar'])
    shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
    shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')
    shared.gradio['truncation_length'] = gr.Slider(value=None, minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
    shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='In addition to the defaults. Written between "" and separated by commas.', placeholder='"\\n", "\\nYou:"')
    shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
    shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='Activate text streaming')
    shared.gradio['character_menu'] = gr.Dropdown(value='None', choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.', elem_classes='slim-dropdown')
    shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
    shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
    shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=5, label='Greeting', elem_classes=['add_scrollbar'])
    shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=10, label='Context', elem_classes=['add_scrollbar'])
    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Instruction template', value='None', info='Change this according to the model/LoRA that you are using. Used in instruct and chat-instruct modes.', elem_classes='slim-dropdown')
    shared.gradio['name1_instruct'] = gr.Textbox(value='', lines=2, label='User string')
    shared.gradio['name2_instruct'] = gr.Textbox(value='', lines=1, label='Bot string')
    shared.gradio['context_instruct'] = gr.Textbox(value='', lines=4, label='Context')
    shared.gradio['turn_template'] = gr.Textbox(value='', lines=1, label='Turn template', info='Used to precisely define the placement of spaces and new line characters in instruction prompts.')
    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Command for chat-instruct mode', info='<|character|> gets replaced by the bot name, and <|prompt|> gets replaced by the regular chat prompt.', elem_classes=['add_scrollbar'])
    shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Preset', elem_classes='slim-dropdown')
    shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
    shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Upload History JSON')
    shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button')
    shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button')
    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button')
    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button')
    shared.gradio['save_chat_history'] = gr.Button(value='Save history')
    shared.gradio['Submit character'] = gr.Button(value='Submit', interactive=False)
    shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON or YAML File')
    shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Profile Picture (optional)')
    shared.gradio['Submit tavern character'] = gr.Button(value='Submit', interactive=False)
    shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG File', elem_id='upload_img_tavern')
    shared.gradio['tavern_json'] = gr.State()
    shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Name', interactive=False)
    shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=4, max_lines=4, label='Description', interactive=False)
    shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=None)
    shared.gradio['send_instruction_to_default'] = gr.Button('Send to default', elem_classes=['small-button'])
    shared.gradio['send_instruction_to_notebook'] = gr.Button('Send to notebook', elem_classes=['small-button'])
    shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('Send to negative prompt', elem_classes=['small-button'])
    shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button')
    shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button')
    shared.gradio['textbox-default'] = gr.Textbox(value='', lines=27, label='Input', elem_classes=['textbox_default', 'add_scrollbar'])
    shared.gradio['token-counter-default'] = gr.HTML(value="<span>0</span>", elem_classes=["token-counter", "default-token-counter"])
    shared.gradio['output_textbox'] = gr.Textbox(lines=27, label='Output', elem_id='textbox-default', elem_classes=['textbox_default_output', 'add_scrollbar'])
    shared.gradio['prompt_menu-default'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
    shared.gradio['save_prompt-default'] = gr.Button('💾', elem_classes='refresh-button')
    shared.gradio['delete_prompt-default'] = gr.Button('🗑️', elem_classes='refresh-button')
    
    with gr.Tab("Session", elem_id="session-tab"):
        with gr.Row():
            with gr.Column():
                shared.gradio['reset_interface'] = gr.Button("Apply flags/extensions and restart")
                with gr.Row():
                    shared.gradio['toggle_dark_mode'] = gr.Button('Toggle 💡')
                    shared.gradio['save_settings'] = gr.Button('Save UI defaults to settings.yaml')

                with gr.Row():
                    with gr.Column():
                        shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Available extensions", info='Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                    with gr.Column():
                        shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=get_boolean_arguments(), value=get_boolean_arguments(active=True), label="Boolean command-line flags", elem_classes='checkboxgroup-table')

            with gr.Column():
                if not shared.args.multi_user:
                    shared.gradio['save_session'] = gr.Button('Save session')
                    shared.gradio['load_session'] = gr.File(type='binary', file_types=['.json'], label="Upload Session JSON")

                extension_name = gr.Textbox(lines=1, label='Install or update an extension', info='Enter the GitHub URL below and press Enter. For a list of extensions, see: https://github.com/oobabooga/text-generation-webui-extensions ⚠️  WARNING ⚠️ : extensions can execute arbitrary code. Make sure to inspect their source code before activating them.')
                extension_status = gr.Markdown()

        extension_name.submit(
            clone_or_pull_repository, extension_name, extension_status, show_progress=False).then(
            lambda: gr.update(choices=utils.get_available_extensions(), value=shared.args.extensions), None, gradio('extensions_menu'))

        # Reset interface event
        shared.gradio['reset_interface'].click(
            set_interface_arguments, gradio('extensions_menu', 'bool_menu'), None).then(
            lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;padding-top:20%;margin:0;height:100vh;color:lightgray;text-align:center;background:var(--body-background-fill)">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        shared.gradio['toggle_dark_mode'].click(lambda: None, None, None, _js='() => {document.getElementsByTagName("body")[0].classList.toggle("dark")}')
        shared.gradio['save_settings'].click(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            ui.save_settings, gradio('interface_state', 'preset_menu', 'instruction_template', 'extensions_menu', 'show_controls'), gradio('save_contents')).then(
            lambda: './', None, gradio('save_root')).then(
            lambda: 'settings.yaml', None, gradio('save_filename')).then(
            lambda: gr.update(visible=True), None, gradio('file_saver'))


def set_interface_arguments(extensions, bool_active):
    shared.args.extensions = extensions

    bool_list = get_boolean_arguments()

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True


def get_boolean_arguments(active=False):
    exclude = ["default", "notebook", "chat"]

    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in exclude + ui.list_model_elements()])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]

    if active:
        return bool_active
    else:
        return bool_list
