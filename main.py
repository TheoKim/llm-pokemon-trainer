import asyncio
import os
import subprocess
import json
import re
import random
from dotenv import load_dotenv
from poke_env.player import Player
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.data import GenData
from poke_env.environment import Status, Field, SideCondition, Weather, PokemonType, MoveCategory, Effect

# --- Environment and Warnings Setup ---
os.environ["MallocStackLogging"] = "0"
load_dotenv()

class LocalLLMPlayer(Player):
    def __init__(self, model="llama", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.message_history = []
        self.initial_prompt_sent = False
        self.last_move = None
        self.just_switched = False

    # --- LLM Helper Functions ---

    def initialize_prompt(self):
        if not self.initial_prompt_sent:
            self.message_history.append({
                "role": "system",
                "content": (
                    "You are a master Pokémon battler. Win the current battle by making smart decisions."
                    "Analyze the current situation: your active Pokémon, the opponent's Pokémon, and available moves."
                    "Provide only the single, lowercase name of the move or switch you want to make. Do not provide any explanation or reasoning." 

                )
            })
            print("[LLM LOG] Initial system prompt configured.")
            self.initial_prompt_sent = True

    def reset_and_initialize_prompt(self):
        # Clears the message history and re-initializes the system prompt.
        self.message_history = []
        self.initial_prompt_sent = False
        self.initialize_prompt()

    def query_ollama(self):
        try:
            prompt_input = json.dumps({"messages": self.message_history})
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt_input.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"[OLLAMA ERROR] Subprocess failed with code {result.returncode}: {result.stderr.decode()}")
                return None
            return result.stdout.decode().strip().lower()
        except FileNotFoundError:
            print("[FATAL ERROR] 'ollama' command not found. Make sure Ollama is installed and in your system's PATH.")
            return None
        except Exception as e:
            print(f"[OLLAMA ERROR] An exception occurred while querying Ollama: {e}")
            return None

    def sanitize_model_response(self, text):
        return re.sub(r"[^a-z-]", "", text.strip().lower())
    
    # --- Building the game state ---

    def get_status_string(self, pokemon):
        # Converts the Status enum to a human-readable (or in this case, LLM-readable) string.
        if pokemon.status is None:
            return "Healthy"
        # The .name attribute of an enum member gives its name as a string (e.g., "BRN")
        return pokemon.status.name.capitalize()

    def get_weather_string(self, battle):
            # This function is now robust against battle.weather being a dict or an enum.
            if not battle.weather:
                return "No weather"
            
            weather = battle.weather
            # If weather is a dictionary, get the actual Weather enum from its keys
            if isinstance(weather, dict):
                weather = list(weather.keys())[0]

            return weather.name.replace('_', ' ').title()

    def get_field_conditions_string(self, battle):
        if not battle.fields:
            return "None"
        return ", ".join([f.name.replace('_', ' ').title() for f in battle.fields])

    def get_side_conditions_string(self, battle, on_my_side=True):
        conditions_dict = battle.side_conditions if on_my_side else battle.opponent_side_conditions
        if not conditions_dict:
            return "None"
        
        condition_strings = []
        for condition, value in conditions_dict.items():
            name = condition.name.replace('_', ' ').title()
            if condition in [SideCondition.SPIKES, SideCondition.TOXIC_SPIKES]:
                condition_strings.append(f"{name} (x{value})")
            else:
                condition_strings.append(name)
        return ", ".join(condition_strings)
    
    def build_turn_prompt(self, battle, legal_actions, damage_info):
        my_status = self.get_status_string(battle.active_pokemon)
        opponent_status = self.get_status_string(battle.opponent_active_pokemon)

        conditions_summary = (
            f"--- Battle State ---\n"
            f"Weather: {self.get_weather_string(battle)}\n"
            f"Field Effects: {self.get_field_conditions_string(battle)}\n"
            f"Your Side: {self.get_side_conditions_string(battle, on_my_side=True)}\n"
            f"Opponent's Side: {self.get_side_conditions_string(battle, on_my_side=False)}\n"
        )

        formatted_actions = []
        for action in legal_actions:
            if action.startswith("switch-"):
                formatted_actions.append(action)
            else:
                info = damage_info.get(action)
                if info:
                    details = []
                    # Add Expected Damage first if it's a damaging move
                    if info.get('expected_damage', 0) > 0:
                        details.append(f"Expected Damage: {info['expected_damage']:.1f}")
                    
                    # Add qualitative descriptors
                    if info.get('stab'): details.append("STAB")
                    if info.get('multiplier', 1) > 1: details.append(f"{info['multiplier']:.1f}x Super Effective")
                    elif info.get('multiplier', 1) < 1 and info.get('multiplier', 1) > 0: details.append(f"{info['multiplier']:.1f}x Not Very Effective")
                    elif info.get('multiplier') == 0: details.append("No Effect")
                    if info.get('priority', 0) > 0: details.append("Priority")
                    
                    if details:
                        formatted_actions.append(f"{action} ({', '.join(details)})")
                    else:
                        formatted_actions.append(action) # Append status moves without details
                else:
                    formatted_actions.append(action)

        # Different prompts depending on situations:
        if not battle.available_moves and battle.available_switches:
            # If LLM is forced to choose a new Pokémon:
            prompt = (
                f"You MUST choose a replacement from ONLY below list.\n"
                f"Opponent's active Pokémon: {battle.opponent_active_pokemon.species} (HP: {battle.opponent_active_pokemon.current_hp_fraction * 100:.1f}%) (Status: {opponent_status})\n"
                f"Your ONLY available team members are: {legal_actions}\n"
                f"switch to a Pokémon that has super effective moves against {battle.opponent_active_pokemon.species} or if {battle.opponent_active_pokemon.species} has moves that are not very effective against {battle.active_pokemon.species}.\n"
                "Any other Pokémon not in the list has fainted or is unavailable. Respond with a single Pokémon name."
                f"{conditions_summary}"

            )
        else:
            # If LLM has to choose a move or optionally switch Pokémon:
            last_move_text = f"Last turn you used '{self.last_move}'." if self.last_move else "This is the first turn for this Pokémon."
            prompt = (
                f"{last_move_text}\n"
                f"DO NOT repeatedly use the move: {self.last_move} UNLESS it is the optimal move to use."
                f"Your active Pokémon: {battle.active_pokemon.species} (HP: {battle.active_pokemon.current_hp_fraction * 100:.1f}%) (Status: {my_status})\n"
                f"Opponent's active Pokémon: {battle.opponent_active_pokemon.species} (HP: {battle.opponent_active_pokemon.current_hp_fraction * 100:.1f}% (Status: {opponent_status})\n"
                "Choose a MOVE or a tactical SWITCH.\n"
                f"Switch if your moves against {battle.opponent_active_pokemon.species} are not very effective AND they have super effective moves against your {battle.active_pokemon.species}.\n"
                "To switch to a different Pokémon, choose an action starting with 'switch-'.\n"
                f"If switching, switch to a Pokémon that has super effective moves against {battle.opponent_active_pokemon.species} or if {battle.opponent_active_pokemon.species} has moves that are not very effective against {battle.active_pokemon.species}.\n"
                f"Your ONLY available actions are: {', '.join(formatted_actions)}\n"
                "If selecting a damaging move, PRIORITIZE moves that are Super Effective and have STAB."
                "Any other action not in the list is invalid. Your response must be a single, exact, lowercase name from the list."
                f"{conditions_summary}"

            )
        return prompt

    # --- Battle Decision Logic ---

    def calculate_move_damages(self, battle):
        # Calculates the expected damage and other key properties for each move.
        gen_data = GenData.from_gen(battle.gen)
        damage_info = {}
        if not battle.available_moves:
            return damage_info

        active_pokemon = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        def get_crit_damage_multiplier(pokemon):
            if pokemon.ability == "sniper":
                if battle.gen >= 6:
                    crit_multiplier = 2.25
                else:
                    crit_multiplier = 3
            else:
                if battle.gen >= 6:
                    crit_multiplier = 1.5
                else:
                    crit_multiplier = 2
            return crit_multiplier

        def get_crit_chance(move, pokemon):
            crit_stage = move.crit_ratio
            if pokemon.ability and "superluck" in pokemon.ability: crit_stage += 1
            if pokemon.item and "scopelens" in pokemon.item: crit_stage += 1
            if battle.gen >= 7: return {0: 1/24, 1: 1/8, 2: 1/2}.get(crit_stage, 1.0)
            if battle.gen == 6: return {0: 1/16, 1: 1/8, 2: 1/2}.get(crit_stage, 1.0)
            return {0: 1/16, 1: 1/8, 2: 1/4, 3: 1/3, 4: 1/2}.get(crit_stage, 0.5)

        for move in battle.available_moves:
            power = move.base_power
            move_type = move.type

            # Handle moves with variable power or special effects
            if move.id == 'weatherball':
                if battle.weather in [Weather.SUNNYDAY, Weather.DESOLATELAND]:
                    power = 100
                    move_type = PokemonType.FIRE
                elif battle.weather in [Weather.RAINDANCE, Weather.PRIMORDIALSEA]:
                    power = 100
                    move_type = PokemonType.WATER
                elif battle.weather == Weather.SANDSTORM:
                    power = 100
                    move_type = PokemonType.ROCK
                elif battle.weather == Weather.SNOW:
                    power = 100
                    move_type = PokemonType.ICE
            
            elif move.id in ['return', 'frustration']:
                power = 102 # Assume max happiness/frustration

            elif move.id in ['heavyslam', 'heatcrash']:
                if opponent.weight and active_pokemon.weight:
                    ratio = opponent.weight / active_pokemon.weight
                    if ratio < 0.2: power = 120
                    elif ratio < 0.25: power = 100
                    elif ratio < 1/3: power = 80
                    elif ratio < 0.5: power = 60
                    else: power = 40
                else:
                    power = 100
                
            # --- Effectiveness Calculation (Handles Immunities First) ---
            effectiveness = opponent.damage_multiplier(move)
            
            def get_defensive_modifier(ability):
                mod = 1.0
                # Immunities
                if (ability == "dryskin" and move_type == PokemonType.WATER): return 0
                if (ability == "flashfire" and move_type == PokemonType.FIRE): return 0
                if (ability == "levitate" and move_type == PokemonType.GROUND): return 0
                if (ability in ["lightningrod", "motordrive", "voltabsorb"] and move_type == PokemonType.ELECTRIC): return 0
                if (ability == "sapsipper" and move_type == PokemonType.GRASS): return 0
                if (ability in ["stormdrain", "waterabsorb"]) and move_type == PokemonType.WATER: return 0
                
                # Damage reduction
                if (ability in ["filter", "solidrock", "prismarmor"]) and effectiveness > 1: mod *= 0.75
                if (ability == "fluffy" and 'contact' in move.flags): mod *= 0.5
                if (ability == "furcoat" and move.category == MoveCategory.PHYSICAL): mod *= 0.5
                if (ability == "heatproof" and move_type == PokemonType.FIRE): mod *= 0.5
                if (ability == "icescales" and move.category == MoveCategory.SPECIAL): mod *= 0.5
                if (ability == "multiscale" and opponent.current_hp_fraction == 1): mod *= 0.5
                if (ability == "punkrock" and 'sound' in move.flags): mod *= 0.5
                if (ability == "purifyingsalt" and move_type == PokemonType.GHOST): mod *= 0.5
                if (ability == "thickfat" and move_type in [PokemonType.FIRE, PokemonType.ICE]): mod *= 0.5
                if (ability == "waterbubble" and move_type == PokemonType.FIRE): mod *= 0.5
                
                # Damage increase
                if (ability == "dryskin" and move_type == PokemonType.FIRE): mod *= 1.25
                if (ability == "fluffy" and move_type == PokemonType.FIRE): mod *= 2.0
                return mod
            
            if opponent.ability:
                effectiveness *= get_defensive_modifier(opponent.ability)
            else:
                worst_case_modifier = 1.0
                for possible_ability in opponent.possible_abilities:
                    modifier = get_defensive_modifier(possible_ability)
                    if modifier < worst_case_modifier:
                        worst_case_modifier = modifier
                effectiveness *= worst_case_modifier

            if active_pokemon.ability == "tintedlens" and effectiveness < 1:
                effectiveness *= 2

            # --- Handle Status Moves and Immunities ---
            is_stab = move_type in active_pokemon.types
            if move.category == MoveCategory.STATUS or effectiveness == 0:
                damage_info[move.id] = { "expected_damage": 0, "stab": is_stab, "multiplier": effectiveness, "priority": move.priority }
                continue
            
            # --- Continue with full calculation for damaging moves ---
            power = move.base_power
            attacker_ability = active_pokemon.ability
            
            if attacker_ability == "adaptability" and is_stab: power *= 2.0
            elif is_stab: power *= 1.5

            # --- Attacker's Ability Power Boosts ---
            if attacker_ability == "aerilate" and move_type == PokemonType.NORMAL: power *= 1.2
            elif attacker_ability == "analytic" and (battle.turn > 1 and self.last_move == move.id): power *= 1.3
            elif attacker_ability == "blaze" and active_pokemon.current_hp_fraction <= 1/3 and move_type == PokemonType.FIRE: power *= 1.5
            elif attacker_ability == "darkaura": power *= 1.33
            elif attacker_ability == "fairyaura": power *= 1.33
            elif attacker_ability == "flareboost" and active_pokemon.status == Status.BRN and move.category == MoveCategory.SPECIAL: power *= 1.5
            elif attacker_ability == "guts" and active_pokemon.status is not None and move.category == MoveCategory.PHYSICAL: power *= 1.5
            elif attacker_ability == "ironfist" and move.flags and 'punch' in move.flags: power *= 1.2
            elif attacker_ability == "megalauncher" and move.flags and 'pulse' in move.flags: power *= 1.5
            elif attacker_ability == "overgrow" and active_pokemon.current_hp_fraction <= 1/3 and move_type == PokemonType.GRASS: power *= 1.5
            elif attacker_ability == "pixilate" and move_type == PokemonType.NORMAL: power *= 1.2
            elif attacker_ability == "punkrock" and move.flags and 'sound' in move.flags: power *= 1.3
            elif attacker_ability == "reckless" and move.recoil > 0: power *= 1.2
            elif attacker_ability == "refrigerate" and move_type == PokemonType.NORMAL: power *= 1.2
            elif attacker_ability == "sandforce" and battle.weather == Weather.SANDSTORM and move_type in [PokemonType.ROCK, PokemonType.GROUND, PokemonType.STEEL]: power *= 1.3
            elif attacker_ability == "solarpower" and battle.weather in [Weather.SUNNYDAY, Weather.DESOLATELAND] and move.category == MoveCategory.SPECIAL: power *= 1.5
            elif attacker_ability in ["steelworker", "steelyspirit"] and move_type == PokemonType.STEEL: power *= 1.5
            elif attacker_ability == "strongjaw" and move.flags and 'bite' in move.flags: power *= 1.5
            elif attacker_ability == "swarm" and active_pokemon.current_hp_fraction <= 1/3 and move_type == PokemonType.BUG: power *= 1.5
            elif attacker_ability == "technician" and move.base_power <= 60: power *= 1.5
            elif attacker_ability == "torrent" and active_pokemon.current_hp_fraction <= 1/3 and move_type == PokemonType.WATER: power *= 1.5
            elif attacker_ability == "toughclaws" and move.flags and 'contact' in move.flags: power *= 1.3
            elif attacker_ability == "toxicboost" and active_pokemon.status in [Status.PSN, Status.TOX] and move.category == MoveCategory.PHYSICAL: power *= 1.5
            elif attacker_ability == "transistor" and move_type == PokemonType.ELECTRIC: power *= 1.3
            elif attacker_ability == "waterbubble" and move_type == PokemonType.WATER: power *= 2.0

            # --- Weather & Terrain Modifiers ---
            is_grounded = PokemonType.FLYING not in active_pokemon.types and attacker_ability != 'levitate'
            if battle.weather in [Weather.SUNNYDAY, Weather.DESOLATELAND] and move_type == PokemonType.FIRE: power *= 1.5
            if battle.weather in [Weather.SUNNYDAY, Weather.DESOLATELAND] and move_type == PokemonType.WATER: power *= 0.5
            if battle.weather in [Weather.RAINDANCE, Weather.PRIMORDIALSEA] and move_type == PokemonType.WATER: power *= 1.5
            if battle.weather in [Weather.RAINDANCE, Weather.PRIMORDIALSEA] and move_type == PokemonType.FIRE: power *= 0.5
            if Field.ELECTRIC_TERRAIN in battle.fields and move_type == PokemonType.ELECTRIC and is_grounded: power *= 1.3
            if Field.GRASSY_TERRAIN in battle.fields and move_type == PokemonType.GRASS and is_grounded: power *= 1.3
            if Field.PSYCHIC_TERRAIN in battle.fields and move_type == PokemonType.PSYCHIC and is_grounded: power *= 1.3

            # --- Final Damage Modifier (Burn, Screens, etc.) ---
            final_modifier = 1.0
            if active_pokemon.status == Status.BRN and move.category == MoveCategory.PHYSICAL and attacker_ability != "guts": final_modifier *= 0.5
            if SideCondition.REFLECT in battle.opponent_side_conditions and move.category == MoveCategory.PHYSICAL: final_modifier *= 0.5
            if SideCondition.LIGHT_SCREEN in battle.opponent_side_conditions and move.category == MoveCategory.SPECIAL: final_modifier *= 0.5
            if battle.weather == Weather.SANDSTORM and PokemonType.ROCK in opponent.types and move.category == MoveCategory.SPECIAL and battle.gen >= 4:
                final_modifier *= (1 / 1.5)
            if battle.weather == Weather.SNOW and PokemonType.ICE in opponent.types and move.category == MoveCategory.PHYSICAL and battle.gen >= 9:
                final_modifier *= (1 / 1.5)

            # --- Because Freeze-Dry has to be a special child ---
            if move.id == 'freezedry':
                effectiveness = 1.0
                for opponent_type in opponent.types:
                    if opponent_type:
                        if opponent_type == PokemonType.WATER:
                            effectiveness *= 2
                        else:
                            effectiveness *= opponent_type.damage_multiplier(
                                PokemonType.ICE, type_chart=gen_data.type_chart
                            )

            # --- Final Calculation ---
            base_damage = power * effectiveness * final_modifier
            crit_chance = get_crit_chance(move, active_pokemon)
            crit_multiplier = get_crit_damage_multiplier(active_pokemon)
            crit_damage_modifier = 1 + (crit_chance * (crit_multiplier - 1))
            
            expected_damage = (move.accuracy / 100.0) * base_damage * crit_damage_modifier
            
            damage_info[move.id] = {
                "expected_damage": expected_damage,
                "stab": is_stab,
                "multiplier": effectiveness,
                "priority": move.priority
            }
        
        return damage_info
    
    def filter_suboptimal_moves(self, battle, legal_actions, damage_map):
        # Removes currently tactically poor moves.

        filtered_actions = legal_actions.copy()
        active_pokemon = battle.active_pokemon
        current_opponent = battle.opponent_active_pokemon

        # These are the most important circumstances to consider on any given turn:
        # Can we KO our opponent on this turn, or are we in danger of being KO'ed?

        # Rule 0: Mortal peril
        filtered_actions = self.mortal_peril_alert(battle, filtered_actions, damage_map)

        # Rule 0.5: Step on throats
        if current_opponent.current_hp_fraction < 0.5:   
            filtered_actions = self.step_on_throat(battle, filtered_actions, damage_map)
            
        # Rule 1: Don't use healing moves if HP is high
        if active_pokemon.current_hp_fraction >= 0.66:
            healing_moves = [
                'healorder', 'milkdrink', 'moonlight', 'morningsun', 'recover', 'rest', 
                'roost', 'shoreup', 'slackoff', 'softboiled', 'swallow', 'synthesis'
            ]
            for move in healing_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removed '{move}' (Current HP >= 0.66).")
                    filtered_actions.remove(move)

        # Rule 2: Don't use hazard-setting or screen moves if they are already active
        opp_side_conditions = battle.opponent_side_conditions
        my_side_conditions = battle.side_conditions
        if SideCondition.STEALTH_ROCK in opp_side_conditions and 'stealthrock' in filtered_actions:
            print(f"[GAME LOG] Removed 'stealthrock' (Already active).")
            filtered_actions.remove('stealthrock')
        if opp_side_conditions.get(SideCondition.SPIKES, 0) >= 3 and 'spikes' in filtered_actions:
            print(f"[GAME LOG] Removed 'spikes' (Max stacks).")
            filtered_actions.remove('spikes')
        if opp_side_conditions.get(SideCondition.TOXIC_SPIKES, 0) >= 2 and 'toxicspikes' in filtered_actions:
            print(f"[GAME LOG] Removed 'toxicspikes' (Max stacks).")
            filtered_actions.remove('toxicspikes')
        if SideCondition.STICKY_WEB in opp_side_conditions and 'stickyweb' in filtered_actions:
            print(f"[GAME LOG] Removed 'stickyweb' (Already active).")
            filtered_actions.remove('stickyweb')
        if SideCondition.REFLECT in my_side_conditions and 'reflect' in filtered_actions:
            print(f"[GAME LOG] Removed 'reflect' (Already active).")
            filtered_actions.remove('reflect')
        if SideCondition.LIGHT_SCREEN in my_side_conditions and 'lightscreen' in filtered_actions:
            print(f"[GAME LOG] Removed 'lightscreen' (Already active).")
            filtered_actions.remove('lightscreen')
        if SideCondition.AURORA_VEIL in my_side_conditions and 'auroraveil' in filtered_actions:
            print(f"[GAME LOG] Removed 'auroraveil' (Already active).")
            filtered_actions.remove('auroraveil')        
        if SideCondition.TAILWIND in my_side_conditions and 'tailwind' in filtered_actions:
            print(f"[GAME LOG] Removed 'tailwind' (Already active).")
            filtered_actions.remove('tailwind')  
        if Field.TRICK_ROOM in battle.fields and 'trickroom' in filtered_actions:
            print("[GAME LOG] Removing 'trickroom' (already active).")
            filtered_actions.remove('trickroom')

        # Rule 3: Don't use hazard-clearing moves if there are no hazards on our side
        hazard_removers = ['rapidspin', 'mortalspin', 'defog']
        my_side_hazards = any(c in [SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB] for c in battle.side_conditions)
        if not my_side_hazards:
            for move in hazard_removers:
                if move in filtered_actions and move != 'defog': # Defog has other uses
                    print(f"[GAME LOG] Removed '{move}' (No hazards to clear).")
                    filtered_actions.remove(move)
            # Special case for Defog: also check opponent hazards/screens
            if 'defog' in filtered_actions and not any(battle.opponent_side_conditions.values()):
                 print(f"[GAME LOG] Removed 'defog' (No hazards/screens on either side).")
                 filtered_actions.remove('defog')

        # Rule 4: Don't use setup moves if stats are already maxed        
        boosts = active_pokemon.boosts

        # Single-stat boosts
        if boosts.get('atk', 0) >= 2 and 'swordsdance' in filtered_actions:
            print("[GAME LOG] Removing 'swordsdance' (Attack already boosted).")
            filtered_actions.remove('swordsdance')
        if boosts.get('spa', 0) >= 2 and 'nastyplot' in filtered_actions:
            print("[GAME LOG] Removing 'nastyplot' (Sp. Atk already boosted).")
            filtered_actions.remove('nastyplot')
        if boosts.get('def', 0) >= 2:
            if 'irondefense' in filtered_actions: filtered_actions.remove('irondefense')
            if 'acidarmor' in filtered_actions: filtered_actions.remove('acidarmor')
            print("[GAME LOG] Removing Def-boosting move (Defense already boosted).")
        if boosts.get('spd', 0) >= 2 and 'amnesia' in filtered_actions:
            print("[GAME LOG] Removing 'amnesia' (Sp. Def already boosted).")
            filtered_actions.remove('amnesia')
        if boosts.get('spe', 0) >= 2:
            if 'agility' in filtered_actions: filtered_actions.remove('agility')
            if 'rockpolish' in filtered_actions: filtered_actions.remove('rockpolish')
            print("[GAME LOG] Removing Spe-boosting move (Speed already boosted).")

        # Multi-stat boosts
        if boosts.get('atk', 0) >= 1 and boosts.get('def', 0) >= 1:
             if 'bulkup' in filtered_actions: filtered_actions.remove('bulkup')
             if 'coil' in filtered_actions: filtered_actions.remove('coil')
             print("[GAME LOG] Removing Atk/Def boosting move (Stats already boosted).")
        if boosts.get('atk', 0) >= 1 and boosts.get('spe', 0) >= 1:
            if 'dragondance' in filtered_actions: filtered_actions.remove('dragondance')
            print("[GAME LOG] Removing 'dragondance' (Stats already boosted).")

        if boosts.get('spa', 0) >= 1 and boosts.get('spd', 0) >= 1:
            if 'calmmind' in filtered_actions: filtered_actions.remove('calmmind')
            print("[GAME LOG] Removing 'calmmind' (Stats already boosted).")

        if boosts.get('spa', 0) >= 1 and boosts.get('spd', 0) >= 1 and boosts.get('spe', 0) >= 1:
            if 'quiverdance' in filtered_actions: filtered_actions.remove('quiverdance')
            print("[GAME LOG] Removing 'quiverdance' (Stats already boosted).")

        # Special cases like Shell Smash
        if 'shellsmash' in filtered_actions and (boosts.get('atk', 0) >= 2 or boosts.get('spa', 0) >= 2) and boosts.get('spe', 0) >= 2:
            print("[GAME LOG] Removing 'shellsmash' (Stats already boosted).")
            filtered_actions.remove('shellsmash')
        
        # Curse for non-Ghost types
        if 'curse' in filtered_actions and PokemonType.GHOST not in active_pokemon.types:
            if boosts.get('atk', 0) >= 1 and boosts.get('def', 0) >= 1:
                print("[GAME LOG] Removing 'curse' (Stats already boosted).")
                filtered_actions.remove('curse')

        # Rule 5: If locked into a bad move, only allow switching
        if len(battle.available_moves) == 1:
            locked_move = battle.available_moves[0]
            effectiveness = battle.opponent_active_pokemon.damage_multiplier(locked_move)
            if effectiveness < 1:
                print(f"[GAME LOG] Locked into ineffective move '{locked_move.id}'. Forcing switch.")

                # First, get all possible switches as Pokemon objects
                all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
                # Then, get the list of VIABLE Pokemon objects
                viable_switches = self.filter_suboptimal_switches(battle, all_possible_switches)
                
                # If all switches are suboptimal, it's better to stay in and attack
                if not viable_switches:
                    print("[GAME LOG] All switches are suboptimal, not forcing a switch.")
                    return filtered_actions

                return viable_switches
        
        # Rule 6: Don't use moves that have 0 expected damage (immunities)
        if damage_map:
            for move_id, info in damage_map.items():
                if info.get("expected_damage", 0) == 0:
                    # Check if it's a damaging move by looking at the move object
                    move_obj = next((m for m in battle.available_moves if m.id == move_id), None)
                    if move_obj and move_obj.category != MoveCategory.STATUS:
                        if move_id in filtered_actions:
                            print(f"[GAME LOG] Removed immune attack '{move_id}'.")
                            filtered_actions.remove(move_id)
        
        # Rule 7: Don't use very ineffective moves (very low damage)
        if damage_map:
            for move_id, info in damage_map.items():
                if info.get("expected_damage", 0) <= 0.5:
                    # Check if it's a damaging move by looking at the move object
                    move_obj = next((m for m in battle.available_moves if m.id == move_id), None)
                    if move_obj and move_obj.category != MoveCategory.STATUS:
                        if move_id in filtered_actions:
                            print(f"[GAME LOG] Removed low damage attack '{move_id}'.")
                            filtered_actions.remove(move_id)

        # Rule 8: Only put 1 enemy Pokémon to sleep to prevent sleep clause.
        is_opponent_asleep = any(p.status == Status.SLP for p in battle.opponent_team.values())
        if is_opponent_asleep:
            sleep_moves = ['darkvoid', 'grasswhistle', 'hypnosis', 'lovelykiss', 'sing', 'sleeppowder', 'spore', 'yawn']
            for move in sleep_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (Prevent Sleep Clause trigger).")
                    filtered_actions.remove(move)

        # Rule 9: If X effect is active, do not repeat it unnecessarily.
        if Effect.TAUNT in current_opponent.effects and 'taunt' in filtered_actions:
            print("[GAME LOG] Removing 'taunt' (Opponent already taunted).")
            filtered_actions.remove('taunt')
        if Effect.ENCORE in current_opponent.effects and 'encore' in filtered_actions:
            print("[GAME LOG] Removing 'encore' (Opponent already encored).")
            filtered_actions.remove('encore')
        if Effect.LEECH_SEED in current_opponent.effects and 'leechseed' in filtered_actions:
            print("[GAME LOG] Removing 'leechseed' (Opponent already seeded).")
            filtered_actions.remove('leechseed')

        trapping_effects = [Effect.BIND, Effect.CLAMP, Effect.FIRE_SPIN, Effect.INFESTATION, Effect.SAND_TOMB, Effect.SNAP_TRAP, Effect.THUNDER_CAGE, Effect.WHIRLPOOL, Effect.WRAP]
        is_opponent_trapped = any(effect in current_opponent.effects for effect in trapping_effects)
        if is_opponent_trapped:
            trapping_moves = ['bind', 'clamp', 'firespin', 'infestation', 'sandtomb', 'snaptrap', 'thundercage', 'whirlpool', 'wrap']
            for move in trapping_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing trapping move '{move}' (Opponent already trapped).")
                    filtered_actions.remove(move)

        # Rule 10: If Pokemon is asleep and has sleep timer = 0 or 1 remove all moves except sleep talk

        if active_pokemon.status == Status.SLP:
            # The pokemon is asleep.
            # The status_counter shows remaining turns asleep. If it's 1, it wakes up this turn.
            if active_pokemon.status_counter <= 1:
                # It will remain asleep after this turn.
                if 'sleeptalk' in legal_actions:
                    print("[GAME LOG] Pokémon is asleep. Forcing 'sleeptalk' as the only move.")
                    # Return only 'sleeptalk' and any available switches.
                    filtered_actions = ['sleeptalk'] + [a for a in filtered_actions if a.startswith("switch-")]
                else:
                    # It's asleep but doesn't have 'sleeptalk'. Must switch.
                    print("[GAME LOG] Pokémon is asleep with no 'sleeptalk' move. Forcing switch.")
                    return [action for action in filtered_actions if action.startswith("switch-")]
            else:
                    # It's asleep but sleep turns are 2-3. 
                    print("[GAME LOG] Pokémon is asleep with no 'sleeptalk' move. Forcing switch.")
                    return [action for action in filtered_actions if action.startswith("switch-")]
        else:
            # The pokemon is not asleep, so it cannot use 'sleeptalk'.
            if 'sleeptalk' in filtered_actions:
                print("[GAME LOG] Removing 'sleeptalk' (Pokémon is not asleep).")
                filtered_actions.remove('sleeptalk')
        
        # Rule 11: Don't use Trick if we have NO ITEM or if the enemy has an untrickable item.
        if 'trick' in filtered_actions or 'switcheroo' in filtered_actions:
            if active_pokemon.item is None or (current_opponent.ability and "sticky-hold" in current_opponent.ability):
                if 'trick' in filtered_actions: filtered_actions.remove('trick')
                if 'switcheroo' in filtered_actions: filtered_actions.remove('switcheroo')
                print("[GAME LOG] Removing 'trick'/'switcheroo' (No item or opponent has Sticky Hold).")
        
        # Rule 12: If wish is already up, don't use it again.
        if self.last_move == 'wish' and 'wish' in filtered_actions:
            print("[GAME LOG] Removing 'wish' (Used last turn).")
            filtered_actions.remove('wish')

        # Rule 13: If opponent is already statused, don't select a move to status it again.
        if current_opponent.status is not None:
            status_moves = ['glare', 'poisonpowder', 'poisongas', 'stunspore', 'thunderwave', 'toxic', 'willowisp', 'darkvoid', 'grasswhistle', 'hypnosis', 'lovelykiss', 'sing', 'sleeppowder', 'spore', 'yawn']
            for move in status_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (Opponent already statused).")
                    filtered_actions.remove(move)
        if PokemonType.ELECTRIC in current_opponent.types and battle.gen >= 6:
            if 'thunderwave' in filtered_actions: filtered_actions.remove('thunderwave')
        if PokemonType.FIRE in current_opponent.types:
            if 'willowisp' in filtered_actions: filtered_actions.remove('willowisp')
        if (PokemonType.POISON in current_opponent.types or PokemonType.STEEL in current_opponent.types) and active_pokemon.ability != "corrosion":
            if 'toxic' in filtered_actions: filtered_actions.remove('toxic')
            if 'poisonpowder' in filtered_actions: filtered_actions.remove('poisonpowder')
        if PokemonType.GRASS in current_opponent.types:
            if 'leechseed' in filtered_actions: filtered_actions.remove('leechseed')
                
        # Rule 13.5: If the opponent has an ability making it immune to certain status moves, don't use it.
        possible_abilities = current_opponent.possible_abilities
        if current_opponent.ability:
            possible_abilities = [current_opponent.ability] # If ability is known, only check that one.

        if possible_abilities:
            # Powder move immunities
            if 'overcoat' in possible_abilities or (PokemonType.GRASS in current_opponent.types and battle.gen >= 6):
                powder_moves = ['poisonpowder', 'sleeppowder', 'spore', 'stunspore']
                for move in powder_moves:
                    if move in filtered_actions:
                        print(f"[TACTICAL FILTER] Removing '{move}' (Opponent is immune to powder moves).")
                        filtered_actions.remove(move)

            # Sleep immunities
            if any(ability in ['insomnia', 'vitalsprit'] for ability in possible_abilities):
                sleep_moves = ['darkvoid', 'grasswhistle', 'hypnosis', 'lovelykiss', 'sing', 'sleeppowder', 'spore', 'yawn']
                for move in sleep_moves:
                    if move in filtered_actions:
                        print(f"[TACTICAL FILTER] Removing '{move}' (Opponent is immune to sleep).")
                        filtered_actions.remove(move)

            # Burn immunity
            if 'waterveil' in possible_abilities and 'willowisp' in filtered_actions:
                print("[TACTICAL FILTER] Removing 'willowisp' (Opponent has Water Veil).")
                filtered_actions.remove('willowisp')

            # Paralysis immunity
            if 'limber' in possible_abilities or (PokemonType.ELECTRIC in current_opponent.types and battle.gen >= 6):
                paralysis_moves = ['thunderwave', 'glare', 'nuzzle']
                for move in paralysis_moves:
                    if move in filtered_actions:
                        print(f"[TACTICAL FILTER] Removing '{move}' (Opponent is immune to paralysis).")
                        filtered_actions.remove(move)
            
            # Taunt immunity
            if 'oblivious' in possible_abilities and 'taunt' in filtered_actions and battle.gen >= 6:
                print("[TACTICAL FILTER] Removing 'taunt' (Opponent has Oblivious).")
                filtered_actions.remove('taunt')
                
        # Rule 14: Pain Split
        if 'painsplit' in filtered_actions and active_pokemon.current_hp_fraction >= current_opponent.current_hp_fraction:
            print("[GAME LOG] Removing 'painsplit' (HP >= opponent HP).")
            filtered_actions.remove('painsplit')

        # Rule 15: If primary attacking stat is lowered to -2, only allow switches. Don't stay in, attack, and barely deal damage.
        atk_boost = active_pokemon.boosts.get('atk', 0)
        spa_boost = active_pokemon.boosts.get('spa', 0)
        physical_moves = sum(1 for m in battle.available_moves if m.category == MoveCategory.PHYSICAL)
        special_moves = sum(1 for m in battle.available_moves if m.category == MoveCategory.SPECIAL)

        should_force_switch = False
        if physical_moves > special_moves and atk_boost <= -2:
            print("[GAME LOG] Physical attacker's Attack is too low. Forcing switch.")
            should_force_switch = True
        elif special_moves > physical_moves and spa_boost <= -2:
            print("[GAME LOG] Special attacker's Special Attack is too low. Forcing switch.")
            should_force_switch = True
        elif physical_moves == special_moves and (atk_boost <= -2 or spa_boost <= -2):
            print("[GAME LOG] Mixed attacker's stats are too low. Forcing switch.")
            should_force_switch = True
        
        if should_force_switch:
            # First, get all possible switches
            all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]

            # Then, get the list of VIABLE pokemon objects
            viable_switches = self.filter_suboptimal_switches(battle, all_possible_switches)
            
            if not viable_switches:
                 print("[GAME LOG] All switches are suboptimal, not forcing a switch.")
                 return filtered_actions # Return the original filtered list without forcing a switch

            # Otherwise, return the list of viable switch names
            return viable_switches
            
            
        # Rule 16: If no other Pokemon are alive, do not use switch moves (Teleport, Baton Pass).
        # Or weak switch moves
        if len(battle.team) == 1:
            switch_moves = ['batonpass', 'teleport', 'flipturn', 'voltswitch']
            for move in switch_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (Last Pokémon).")
                    filtered_actions.remove(move)

        # Rule 17: If Protect was used, don't attempt a double protect.
        if self.last_move in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'obstruct', 'burningbulwark', 'silktrap']:
            for move in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'obstruct', 'burningbulwark', 'silktrap']:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (Used last turn).")
                    filtered_actions.remove(move)

        # Rule 18: If weather isn't up, don't use weather related moves
        if battle.weather not in [Weather.SUNNYDAY, Weather.DESOLATELAND]:
            sun_moves = ['solarbeam', 'solarblade']
            for move in sun_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (No sun).")
                    filtered_actions.remove(move)
        
        if battle.weather not in [Weather.RAINDANCE, Weather.PRIMORDIALSEA]:
            rain_moves = ['electroshot']
            for move in rain_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (No rain).")
                    filtered_actions.remove(move)

        if battle.weather not in [Weather.HAIL, Weather.SNOW, Weather.SNOWSCAPE]:
            hail_moves = ['auroraveil']
            for move in hail_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (No hail/snow).")
                    filtered_actions.remove(move)

        if battle.weather in [Weather.RAINDANCE, Weather.PRIMORDIALSEA, Weather.DELTASTREAM, Weather.HAIL, Weather.SANDSTORM, Weather.SNOW, Weather.SNOWSCAPE]:
            sun_healing_moves = ["synthesis", "moonlight", "morningsun"]
            for move in sun_healing_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (Weather reduces healing) ")

        if battle.weather not in [Weather.RAINDANCE, Weather.PRIMORDIALSEA, Weather.DELTASTREAM, Weather.DESOLATELAND, Weather.HAIL, Weather.SANDSTORM, Weather.SNOW, Weather.SNOWSCAPE, Weather.SUNNYDAY]:
            weather_moves = ['weatherball']
            for move in weather_moves:
                if move in filtered_actions:
                    print(f"[GAME LOG] Removing '{move}' (No weather).")
                    filtered_actions.remove(move)

        # Rule 20: If opponent has no item and Knock Off isn't super effective, delete it.
        if 'knockoff' in filtered_actions and not current_opponent.item and damage_map.get('knockoff', {}).get('multiplier', 1) <= 1:
            print("[GAME LOG] Removing 'knockoff' (Opponent has no item and it's not super effective).")
            filtered_actions.remove('knockoff')
                         
        # Rule 21: If opponent is behind a substitue, don't use status moves.
        if Effect.SUBSTITUTE in current_opponent.effects:
            status_moves_to_remove = []
            for move_id in filtered_actions:
                if not move_id.startswith("switch-"):
                    move_obj = next((m for m in battle.available_moves if m.id == move_id), None)
                    if move_obj and move_obj.category == MoveCategory.STATUS:
                        status_moves_to_remove.append(move_id)
            if status_moves_to_remove:
                print(f"[GAME LOG] Removing status moves {status_moves_to_remove} (Opponent has a Substitute).")
                filtered_actions = [a for a in filtered_actions if a not in status_moves_to_remove]

        # Rule 22: If already created a substitute, don't use substitute.
        if Effect.SUBSTITUTE in active_pokemon.effects and 'substitute' in filtered_actions:
            print("[GAME LOG] Removing 'substitute' (Already have a Substitute).")
            filtered_actions.remove('substitute')

        # Rule 23: Prankster vs. Dark-type
        if active_pokemon.ability == 'prankster' and battle.gen >= 7 and PokemonType.DARK in current_opponent.types:
            prankster_moves_to_remove = []
            for move_id in filtered_actions:
                 if not move_id.startswith("switch-"):
                    move_obj = next((m for m in battle.available_moves if m.id == move_id), None)
                    # Prankster only boosts status moves
                    if move_obj and move_obj.category == MoveCategory.STATUS:
                        prankster_moves_to_remove.append(move_id)
            if prankster_moves_to_remove:
                print(f"[GAME LOG] Removing Prankster moves {prankster_moves_to_remove} (vs Dark-type).")
                filtered_actions = [a for a in filtered_actions if a not in prankster_moves_to_remove]

        # Rule 24: If a Pokemon with the ability Truant is loafing, force a switch.
        if active_pokemon.ability == 'truant' and active_pokemon.must_recharge:
            print("[GAME LOG] Truant is active, forcing a switch to avoid giving the opponent a free turn.")
            
            # Create a list of only the available switch actions.
            switch_actions = [action for action in filtered_actions if action.startswith("switch-")]

            if switch_actions:
                # If we can switch, we absolutely should. Return only those options.
                return switch_actions
            else:
                # If there's nothing to switch to (last pokemon), we're stuck.
                # Returning an empty list will cause the pokemon to loaf, which is correct.
                print("[GAME LOG] Truant is active, but no Pokémon are available to switch to.")
                return []
        
        # Rule 25: Don't select moves that have recharge turns (Giga Impact, Hyper Beam).
        recharge_moves = ['hyperbeam', 'gigaimpact', 'rockwrecker', 'frenzyplant', 'blastburn', 'hydrocannon', 'roaroftime', 'eternabeam']
        for move_id in recharge_moves:
            if move_id in filtered_actions:
                filtered_actions.remove(move_id)

        # Rule 26: If locked into a move by Encore or Taunt, check if switching is optimal.
        
        # Scenario 1: Locked into a single move by Encore.
        if Effect.ENCORE in active_pokemon.effects and len(battle.available_moves) == 1:
            locked_move = battle.available_moves[0]
            should_force_switch = False
            
            # A: Is it a damaging move? Check if it's weak or has no effect.
            if locked_move.category != MoveCategory.STATUS:
                if damage_map.get(locked_move.id, {}).get('multiplier', 1) < 1:
                    print(f"[GAME LOG] Encored into an ineffective attack '{locked_move.id}'. Evaluating switch.")
                    should_force_switch = True

            # B: Is it a status move? Check if it's now useless.
            else:
                # Check for useless setup moves (if stats are maxed).
                if locked_move.boosts:
                    is_setup_useless = True
                    for stat, value in locked_move.boosts.items():
                        if active_pokemon.boosts.get(stat, 0) < 6:
                           is_setup_useless = False
                           break
                    if is_setup_useless:
                         print(f"[GAME LOG] Encored into a useless setup move '{locked_move.id}'. Evaluating switch.")
                         should_force_switch = True
                
                # Check for useless hazard moves (if they are already active).
                if locked_move.id == 'stealthrock' and SideCondition.STEALTH_ROCK in opp_side_conditions:
                    should_force_switch = True
                elif locked_move.id == 'spikes' and opp_side_conditions.get(SideCondition.SPIKES, 0) >= 3:
                     should_force_switch = True
                elif locked_move.id == 'stickyweb' and SideCondition.STICKY_WEB in opp_side_conditions:
                     should_force_switch = True
            
            if should_force_switch:
                all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
                viable_switches = self.filter_suboptimal_switches(battle, all_possible_switches)
                if viable_switches:
                    print("[GAME LOG] Forcing an optimal switch.")
                    return viable_switches

        # Scenario 2: Locked into only attacking moves by Taunt.
        if Effect.TAUNT in active_pokemon.effects:
            # Check if all available moves are not very effective or do no damage.
            # 'poke-env' already filters to only show attacking moves when taunted.
            if battle.available_moves:
                all_attacks_are_bad = all(damage_map.get(m.id, {}).get('multiplier', 1) < 1 for m in battle.available_moves)
                if all_attacks_are_bad:
                    print("[GAME LOG] Taunted with only ineffective attacks available. Evaluating switch.")
                    all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
                    viable_switches = self.filter_suboptimal_switches(battle, all_possible_switches)
                    if viable_switches:
                        print("[GAME LOG] Forcing an optimal switch.")
                        return viable_switches
            else:
                 # Taunted, but has no attacking moves at all (e.g. Shuckle, Chansey). Must switch.
                 print("[GAME LOG] Taunted with no attacking moves. Forcing switch.")
                 all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
                 return self.filter_suboptimal_switches(battle, all_possible_switches)

        # Rule 27: Switch out against special walls if you lack physical attacks.
        special_walls = [
            'blissey', 
            'chansey',
        ]
        if current_opponent.species.lower() in special_walls:
            # Check if we have any physical moves available to break through.
            has_physical_attack = any(move.category == MoveCategory.PHYSICAL for move in battle.available_moves)

            if not has_physical_attack:
                print(f"[GAME LOG] Facing special wall '{current_opponent.species}' with no physical attacks. Evaluating switch.")
                
                # Create a list of formatted strings for the switch options.
                all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
                
                # Find a more suitable Pokémon to switch into.
                viable_switches = self.filter_suboptimal_switches(battle, all_possible_switches)
                
                if viable_switches:
                    print("[GAME LOG] Forcing a switch to a more suitable attacker.")
                    return viable_switches
        
        # Rule 29: If Yawn has been used the previous turn, don't use it twice in a row
        if self.last_move == 'yawn' and 'yawn' in filtered_actions:
            print("[GAME LOG] Removing 'yawn' (Used last turn).")
            filtered_actions.remove('yawn')

        # Rule 30: Don't use explosion unless <25% HP AND opponent >75% HP. We want to make positive trades.
        self_ko_moves = ['explosion', 'selfdestruct', 'mistyexplosion']
        for move_id in self_ko_moves:
             if move_id in filtered_actions:
                 # Simple logic: only use if our HP is low (<33%) AND it can KO a high-HP opponent (>60%).
                 if not (active_pokemon.current_hp_fraction < 0.34 and current_opponent.current_hp_fraction > 0.66):
                     print(f"[GAME LOG] Removing self-KO move '{move_id}' (not a favorable trade).")
                     filtered_actions.remove(move_id)

        # Rule 33: Do not use heal bell unless Pokemon are statused. Also do not use if ally teammate or self have ability Soundproof.
        cleric_moves = ['healbell', 'aromatherapy']
        is_team_statused = any(pkmn.status is not None for pkmn in battle.team.values() if not pkmn.fainted)
        for move_id in cleric_moves:
            if move_id in filtered_actions and not is_team_statused:
                 print(f"[GAME LOG] Removing '{move_id}' (no team members are statused).")
                 filtered_actions.remove(move_id)

        # Rule 36: If Trick Room was used on the previous turn, do not use it again.
        if self.last_move == 'trickroom' and 'trickroom' in filtered_actions:
            print("[GAME LOG] Removing 'trickroom' (Used last turn, would reverse the effect).")
            filtered_actions.remove('trickroom')
            
        return filtered_actions

    def filter_suboptimal_switches(self, battle, available_switches):
        # Analyzes the opponent's active Pokemon and removes bad switch-in options.
        # Returns a list of viable switch ins.
        if not available_switches:
            return []
        
        filtered_switches = available_switches.copy()
        opponent = battle.opponent_active_pokemon
        
        switches_to_remove = []

        if opponent.moves:
            # If the opponent has revealed moves, don't switch in Pokemon weak to them.
            for switch_option in filtered_switches:
                # Strip the "switch-" prefix to get a clean name for comparison
                pokemon_name_to_check = switch_option.replace("switch-", "")
                
                pokemon_to_check = next((p for p in battle.available_switches if p.species.lower().replace(' ', '-') == pokemon_name_to_check), None)
                if not pokemon_to_check: continue

                is_weak = any(pokemon_to_check.damage_multiplier(move) > 1 for move in opponent.moves.values() if move.category != MoveCategory.STATUS)
                if is_weak:
                    print(f"[GAME LOG] Removing switch '{pokemon_to_check.species}' (weak to known move).")
                    switches_to_remove.append(switch_option)
        else:
            # Don't switch in Pokemon weak to the opponent's STAB types.
            for switch_option in filtered_switches:
                pokemon_name_to_check = switch_option.replace("switch-", "")

                pokemon_to_check = next((p for p in battle.available_switches if p.species.lower().replace(' ', '-') == pokemon_name_to_check), None)
                if not pokemon_to_check: continue
                
                is_weak_to_stab = any(opponent_type and pokemon_to_check.damage_multiplier(opponent_type) > 1 for opponent_type in opponent.types)
                if is_weak_to_stab:
                    print(f"[GAME LOG] Removing switch '{pokemon_to_check.species}' (weak to opponent's potential STAB).")
                    switches_to_remove.append(switch_option)
        
        # If all switches are filtered out, return the original list to avoid getting stuck
        if len(switches_to_remove) == len(available_switches) and available_switches:
            print("[WARN] All switches were deemed suboptimal. Presenting all options to avoid getting stuck.")
            return available_switches
            
        return [s for s in filtered_switches if s not in switches_to_remove]

    def select_best_damage_move(self, battle):
        # Upon bugs, LLM defaults to the move with the highest calculated expected damage.
        damage_info = self.calculate_move_damages(battle)
        
        best_move_id = None
        max_damage = -1

        for move_id, info in damage_info.items():
            # Directly use the calculated expected_damage for comparison
            if info.get('expected_damage', -1) > max_damage:
                max_damage = info['expected_damage']
                best_move_id = move_id

        if best_move_id:
            best_move = next((m for m in battle.available_moves if m.id == best_move_id), None)
            print(f"[GAME LOG] WARNING: LLM failed. Choosing best damage move: {best_move.id} (Damage: {max_damage:.1f})")
            self.last_move = best_move.id
            return self.create_order(best_move)
        else:
            print("[GAME LOG] WARNING: No damaging moves found, choosing random move.")
            return self.choose_random_move(battle)
    
    def choose_random_switch(self, battle):
        if battle.available_switches:
            chosen_switch = random.choice(battle.available_switches)
            print(f"[GAME LOG] Choosing random switch: {chosen_switch.species}")
            self.last_move = None
            self.reset_and_initialize_prompt() # Reset memory on random switch
            return self.create_order(chosen_switch)
        else:
            print("[WARN] No available switches found during random selection. Defaulting to random move.")
            return self.select_best_damage_move(battle)

    async def wait_for_moves(self, battle):
        # Allows for pokemon movesets to load, regardless of how long in-game animations are.
        time_waited = 0
        wait_interval = 0.1
        max_wait = 3.0

        while not battle.available_moves and not battle.force_switch and time_waited < max_wait:
            await asyncio.sleep(wait_interval)
            time_waited += wait_interval
        
        if not battle.available_moves:
            print(f"[GAME LOG] WARNING: Waited {max_wait}s but still no moves available.")
            
    def get_effective_speed(self, battle):
        # Calculates and returns the effective speed of both active Pokemon.
        # Takes into account stats, boosts, paralysis, and Tailwind.

        my_pokemon = battle.active_pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        
        def calculate_speed(pokemon, side_conditions):
            speed = pokemon.stats['spe']
            boost = pokemon.boosts.get('spe', 0)
            if boost > 0: speed *= (2 + boost) / 2
            elif boost < 0: speed *= 2 / (2 - boost)

            if pokemon.ability == 'slushrush' and battle.weather in [Weather.HAIL, Weather.SNOW]: speed *= 2
            if pokemon.ability == 'swiftswim' and battle.weather in [Weather.RAINDANCE, Weather.PRIMORDIALSEA]: speed *= 2
            if pokemon.ability == 'chlorophyll' and battle.weather in [Weather.SUNNYDAY, Weather.DESOLATELAND]: speed *= 2
            if pokemon.ability == 'slowstart' and pokemon.turn_count < 5: speed *= 0.5
            if pokemon.status == Status.PAR:
                speed *= 0.5 if battle.gen >= 7 else 0.25
            if SideCondition.TAILWIND in side_conditions: speed *= 2

            return speed

        my_speed = calculate_speed(my_pokemon, battle.side_conditions)
        opponent_speed = calculate_speed(opponent_pokemon, battle.opponent_side_conditions)
        
        return my_speed, opponent_speed

    def determine_who_moves_first(self, battle):
        # Determines if the player's pokemon is expected to move first based on speed, ignoring move priority.

        my_speed, opponent_speed = self.get_effective_speed(battle)
        
        # Check if Trick Room is active
        is_trick_room = Field.TRICK_ROOM in battle.fields
        
        if is_trick_room:
            # In Trick Room, the slower Pokemon moves first.
            return my_speed < opponent_speed
        else:
            # Normally, the faster Pokemon moves first.
            return my_speed > opponent_speed
            
    def step_on_throat(self, battle, legal_actions, damage_info):
        # If the opponent is at low HP and we can KO it, this function will
        # restrict the legal actions to only moves that can secure the KO.

        # get_effective_speed just calculates speed stat, with some modifiers.
        # we now have to determine who goes first.
        am_i_faster = self.determine_who_moves_first(battle)

        if not am_i_faster:
            return legal_actions

        ko_moves = []
        for move_id, info in damage_info.items():
            # Check if the move's expected damage is enough to KO
            if info.get('expected_damage', 0) >= 2.5:
                ko_moves.append(move_id)

        # If we found any KOing moves, this is the only option we should consider.
        if ko_moves:
            print(f"[STEP ON THROAT] Prioritizing KO moves: {ko_moves}")
            return ko_moves
            
        return legal_actions

    def mortal_peril_alert(self, battle, legal_actions, damage_info):
        # If the active pokemon is in danger of being KOed by an opponent,
        # this function restricts the available actions to either KOing the opponent
        # or switching to a safer Pokemon.

        # Note that this function only applies to blatantly one-sided matchups:
        # STAB super effective moves (3x damage)
        # Quad super effective coverage moves (4x damage)

        # It's entirely possible to survive super effective stray hits
  
        my_pokemon = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        am_i_faster = self.determine_who_moves_first(battle)

        has_ko_threat = False

        # Check if the opponent has a revealed quad effective coverage move
        if opponent.moves:
            if any(my_pokemon.damage_multiplier(move) == 4 for move in opponent.moves.values()):
                has_ko_threat = True

        # If no moves are revealed, check if their potential STAB moves are super effective
        else:
            if any(my_pokemon.damage_multiplier(t) > 1 for t in opponent.types if t is not None):
                has_ko_threat = True

        # If neither of the above are true: return original list - we are not in mortal peril.
        if not has_ko_threat:
            return legal_actions
        
        all_possible_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
        viable_switch_names = self.filter_suboptimal_switches(battle, all_possible_switches)

        # If opponent has KO threat, can we outspeed it?
        if has_ko_threat:
            if not am_i_faster:
                print(f"[MORTAL PERIL] {my_pokemon.species} is slower and weak to {opponent.species}. Forcing switches only.")
                # Find all viable switches
                return viable_switch_names
            
            else:
                # Find any moves we have that could KO the opponent in return
                ko_moves = [move_id for move_id, info in damage_info.items() if info.get('expected_damage', 0) >= 2]
                if len(ko_moves) == 0:
                    print(f"[MORTAL PERIL] {my_pokemon.species} is faster but cannot OHKO {opponent.species}. Forcing switches only.")
                    return viable_switch_names
                
                else:
                    print(f"[MORTAL PERIL] {my_pokemon.species} is faster, can OHKO {opponent.species}, but is in danger of being KO'ed. Forcing KO moves only.")
                    final_options = ko_moves
                    return final_options
                    
    async def choose_move(self, battle):
        if battle.active_pokemon.must_recharge:
            print("[GAME LOG] Must recharge. Passing turn.")
            return self.create_order(None) # Passing None defaults to passing the turn
                
        if not self.initial_prompt_sent:
            self.initialize_prompt()

        # I have no idea how this block solves the double switch flag issue but it does
        # Plz do not delete <3
        if battle.force_switch:
            if self.just_switched:
                time_waited = 0
                wait_interval = 0.1
                max_wait = 1.0
                while battle.force_switch and time_waited < max_wait:
                    await asyncio.sleep(wait_interval)
                    time_waited += wait_interval
                
                if battle.force_switch:
                    print(f"[GAME LOG] ERROR: Still in force_switch after waiting {time_waited:.1f}s. Passing turn to avoid loop.")
                    return self.choose_random_move(battle)
                else:
                    print(f"[GAME LOG] Loaded moves.")
            else:
                return await self.handle_switch(battle)
        
        # Did we just switch in on the previous turn?
        is_post_switch_turn = self.just_switched

        # Wait for moves to load
        await self.wait_for_moves(battle)
    
        print()
        print(f"[GAME LOG] Turn {battle.turn}: LLM active Pokémon is {battle.active_pokemon.species}")
        print(f"[GAME LOG] Opponent active Pokémon is {battle.opponent_active_pokemon.species}")
        
        damage_map = self.calculate_move_damages(battle)

        # Build the list of available moves and switches from the now-current state.
        legal_moves = [move.id for move in battle.available_moves]
        legal_switches = [f"switch-{p.species.lower().replace(' ', '-')}" for p in battle.available_switches]
        
        filtered_moves = self.filter_suboptimal_moves(battle, legal_moves, damage_map)
        filtered_switches = self.filter_suboptimal_switches(battle, legal_switches)

        # Based on turn context, decide which actions are permissible.
        if is_post_switch_turn:
            print("[INFO] Post-switch turn. Tactical switches are disabled for this turn.")
            legal_actions = filtered_moves
            self.just_switched = False # Reset the flag now that its purpose is served.
        else:
            legal_actions = filtered_moves + filtered_switches

        if not legal_actions:
            print("[WARN] No legal actions available. Choosing best damage move.")
            return self.select_best_damage_move(battle)

        print(f"[GAME LOG] Legal actions: {legal_actions}")

        # Are we forced to switch by our rules or circumstances?
        is_switch_forced_by_filter = all(a.startswith("switch-") for a in legal_actions) and legal_actions
        if is_switch_forced_by_filter:
            print("[GAME LOG] Filter forces a switch. Resetting LLM memory.")
            self.reset_and_initialize_prompt()

        turn_prompt = self.build_turn_prompt(battle, legal_actions, damage_map)
        self.message_history.append({"role": "user", "content": turn_prompt})
        
        max_retries = 3
        for i in range(max_retries):
            model_response = self.query_ollama()
            if model_response:
                cleaned_response = self.sanitize_model_response(model_response)
                if cleaned_response in legal_actions:

                    # If the model chooses to switch
                    if cleaned_response.startswith("switch-"):
                        pokemon_name = cleaned_response.replace("switch-", "")
                        switch_to_use = next((p for p in battle.available_switches if p.species.lower().replace(' ', '-') == pokemon_name), None)
                        if switch_to_use:
                            print(f"[GAME LOG] TACTICAL SWITCH: {switch_to_use.species}")
                            self.reset_and_initialize_prompt()
                            self.last_move = None
                            self.just_switched = True
                            return self.create_order(switch_to_use)                           
                        
                    # If the model chooses a move
                    else:
                        move_to_use = next((m for m in battle.available_moves if m.id == cleaned_response), None)
                        if move_to_use:
                            print(f"[GAME LOG] MOVE CHOICE: {move_to_use.id}")
                            self.message_history.append({"role": "assistant", "content": cleaned_response})
                            self.last_move = move_to_use.id
                            return self.create_order(move_to_use)
                
                print(f"[GAME LOG] ERROR: LLM chose invalid action '{cleaned_response}'. Retry {i+1}/{max_retries}...")
            else:
                print(f"[GAME LOG] ERROR: No response from model. Retry {i+1}/{max_retries}...")
        
        print("[WARN] All LLM retries failed.")
        return self.select_best_damage_move(battle)

    async def handle_switch(self, battle):
        # Get a clean list of switch options (without "switch-" prefix).
        available_switches_names = [p.species.lower().replace(' ', '-') for p in battle.available_switches]
        
        # Only allow viable switches
        available_switches_names = self.filter_suboptimal_switches(battle, available_switches_names)

        if not available_switches_names:
            print("[GAME LOG] FATAL: Forced to switch, but no Pokémon available.")
            return self.choose_random_move(battle)

        # Determine the reason for the forced switch for better logging.
        reason = "was forced out"
        if battle.active_pokemon.fainted:
            reason = "was KNOCKED OUT"
        print(f"[GAME LOG] INVOLUNTARY SWITCH: {battle.active_pokemon.species} {reason}.")
        print(f"[GAME LOG] Available switches: {available_switches_names}")

        self.reset_and_initialize_prompt()

        # Build the specific prompt for this situation.
        prompt = self.build_turn_prompt(battle, available_switches_names, {})
        self.message_history.append({"role": "user", "content": prompt})

        # Query LLM for a decision.
        model_response = self.query_ollama()
        if model_response:
            cleaned_response = self.sanitize_model_response(model_response)
            
            # Check if the LLM's choice is in our list of valid switches.
            if cleaned_response in available_switches_names:
                switch_to_use = next(s for s in battle.available_switches if s.species.lower().replace(' ', '-') == cleaned_response)
                print(f"[GAME LOG] LLM chose replacement: {switch_to_use.species}")
                self.last_move = None
                self.just_switched = True # Flag that a switch occurred.
                return self.create_order(switch_to_use)
        
        print(f"[GAME LOG] WARNING: LLM failed to choose a valid switch from response: '{model_response}'. Choosing randomly.")
        return self.choose_random_switch(battle)
    

async def main():
    try:
        username = os.getenv("SHOWDOWN_USERNAME")
        password = os.getenv("SHOWDOWN_PASSWORD")
        model = os.getenv("MODEL")
        battle_format = os.getenv("BATTLE_FORMAT")
        personal_account = os.getenv("PERSONALACCOUNT")

        if not all([username, password, model]):
            print("Missing environment variables. Please set SHOWDOWN_USERNAME, SHOWDOWN_PASSWORD, and MODEL in your .env file.")
            return
        
        llm_player = LocalLLMPlayer(
            model=model,
            battle_format=battle_format,
            account_configuration=AccountConfiguration(username, password),
            server_configuration=ShowdownServerConfiguration,
        )

        # Challenge a private account
        # await llm_player.accept_challenges(personal_account, 1)

        # Challenge main ladder
        await llm_player.ladder(1)

    except Exception as e:
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())

