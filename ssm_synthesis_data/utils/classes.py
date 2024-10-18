from typing import List, Dict, Optional
import json

def from_dict(data, cls):
    """
    Convert a dictionary to a class instance.
    """
    if isinstance(data, list):
        return [from_dict(item, cls) for item in data]
    elif isinstance(data, dict):
        if cls == Operation:
            conditions = None
            if 'conditions' in data:
                conditions = from_dict(data.pop('conditions'), OperationConditions)
            return cls(**data, conditions=conditions)
        elif cls == OperationConditions:
            heating_temperature = from_dict(data.get('heating_temperature'), OperationValue) if data.get('heating_temperature') else None
            heating_time = from_dict(data.get('heating_time'), OperationValue) if data.get('heating_time') else None
            return cls(
                heating_temperature=heating_temperature,
                heating_time=heating_time,
                heating_atmosphere=data.get('heating_atmosphere'),
                mixing_device=data.get('mixing_device'),
                mixing_media=data.get('mixing_media')
            )
        elif cls == Formula:
            left_side = from_dict(data['left_side'], FormulaPart)
            right_side = from_dict(data['right_side'], FormulaPart)
            return cls(left_side=left_side, right_side=right_side, element_substitution=data['element_substitution'])
        elif cls == Material:
            composition = None
            if 'composition' in data:
                composition = from_dict(data['composition'], Composition)
            return cls(
                material_string=data['material_string'],
                material_formula=data['material_formula'],
                material_name=data['material_name'],
                phase=data.get('phase'),
                is_acronym=data['is_acronym'],
                composition=composition,
                amount_vars=data.get('amount_vars', {}),
                element_vars=data.get('element_vars', {}),
                additives=data.get('additives', []),
                oxygen_deficiency=data.get('oxygen_deficiency')
            )
        elif cls == ReactionEntry:
            reaction = from_dict(data['reaction'], Formula)
            target = from_dict(data['target'], Material)
            precursors = from_dict(data['precursors'], Material)
            operations = from_dict(data['operations'], Operation)
            return cls(
                doi=data['doi'],
                paragraph_string=data['paragraph_string'],
                synthesis_type=data['synthesis_type'],
                reaction_string=data['reaction_string'],
                reaction=reaction,
                targets_string=data['targets_string'],
                target=target,
                precursors=precursors,
                operations=operations
            )
        else:
            return cls(**data)
    else:
        return data

class FormulaPart:
    def __init__(self, amount: str, material: str):
        self.amount = amount
        self.material = material

    def __repr__(self):
        return f"FormulaPart(amount={self.amount!r}, material={self.material!r})"

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
    
    def default(self, o):
        return o.__dict__
    
    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'amount: str': self.amount,
            'material: str': self.material,
        }



class Formula:
    def __init__(self, left_side: List[FormulaPart], right_side: List[FormulaPart], element_substitution: Dict[str, str]):
        self.left_side = left_side
        self.right_side = right_side
        self.element_substitution = element_substitution

    def __repr__(self):
        return (f"Formula(left_side={self.left_side!r}, right_side={self.right_side!r}, "
                f"element_substitution={self.element_substitution!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def default(self, o):
        return o.__dict__

    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'left_side: List[FormulaPart]': self.left_side,
            'right_side: List[FormulaPart]': self.right_side,
            'element_substitution: Dict[str, str]': self.element_substitution
        }


class Composition:
    def __init__(self, formula: str, amount: str, elements: Dict[str, str]):
        self.formula = formula
        self.amount = amount
        self.elements = elements

    def __repr__(self):
        return (f"Composition(formula={self.formula!r}, amount={self.amount!r}, "
                f"elements={self.elements!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
    
    def default(self, o):
        return o.__dict__

    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'formula: str': self.formula,
            'amount: str': self.amount,
            'elements: Dict[str, str]': self.elements
        }


class Material:
    def __init__(self, material_string: str, material_formula: str, material_name: str, phase: Optional[str], is_acronym: bool, composition: List[Composition], amount_vars: Dict[str, List[str]], element_vars: Dict[str, List[str]], additives: List[str], oxygen_deficiency: Optional[str]):
        self.material_string = material_string
        self.material_formula = material_formula
        self.material_name = material_name
        self.phase = phase
        self.is_acronym = is_acronym
        self.composition = composition
        self.amount_vars = amount_vars
        self.element_vars = element_vars
        self.additives = additives
        self.oxygen_deficiency = oxygen_deficiency

    def __repr__(self):
        return (f"Material(material_string={self.material_string!r}, material_formula={self.material_formula!r}, "
                f"material_name={self.material_name!r}, phase={self.phase!r}, is_acronym={self.is_acronym!r}, "
                f"composition={self.composition!r}, amount_vars={self.amount_vars!r}, "
                f"element_vars={self.element_vars!r}, additives={self.additives!r}, "
                f"oxygen_deficiency={self.oxygen_deficiency!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def default(self, o):
        return o.__dict__
    
    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'material_string: str': self.material_string,
            'material_formula: str': self.material_formula,
            'material_name: str': self.material_name,
            'phase: Optional[str]': self.phase if self.phase else None,
            'is_acronym: bool': self.is_acronym,
            'composition: List[Composition]': self.composition,
            'amount_vars:  Dict[str, List[str]]': self.amount_vars,
            'element_vars: Dict[str, List[str]]': self.element_vars,
            'additives: List[str]': self.additives,
            'oxygen_deficiency: Optional[str]': self.oxygen_deficiency if self.oxygen_deficiency else None,
        }


class OperationValue:
    def __init__(self, min_value: float, max_value: float, values: List[float], units: str):
        self.min_value = min_value
        self.max_value = max_value
        self.values = values
        self.units = units

    def __repr__(self):
        return (f"OperationValue(min_value={self.min_value!r}, max_value={self.max_value!r}, "
                f"values={self.values!r}, units={self.units!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
    
    def default(self, o):
        return o.__dict__
   
    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'min_value: float': self.min_value,
            'max_value: float': self.max_value,
            'values: List[float]': self.values,
            'units: str': self.units,
        }

class OperationConditions:
    def __init__(self, heating_temperature: Optional[List[OperationValue]], heating_time: Optional[List[OperationValue]], heating_atmosphere: Optional[str], mixing_device: Optional[str], mixing_media: Optional[str]):
        self.heating_temperature = heating_temperature
        self.heating_time = heating_time
        self.heating_atmosphere = heating_atmosphere
        self.mixing_device = mixing_device
        self.mixing_media = mixing_media

    def __repr__(self):
        return (f"OperationConditions(heating_temperature={self.heating_temperature!r}, "
                f"heating_time={self.heating_time!r}, heating_atmosphere={self.heating_atmosphere!r}, "
                f"mixing_device={self.mixing_device!r}, mixing_media={self.mixing_media!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def default(self, o):
        return o.__dict__

    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'heating_temperature: Optional[List[OperationValue]]': [x.to_dict() for x in self.heating_temperature] if self.heating_temperature else None,
            'heating_time: Optional[List[OperationValue]]': [x.to_dict() for x in self.heating_time]if self.heating_time else None,
            'heating_atmosphere: Optional[str]': self.heating_atmosphere if self.heating_atmosphere else None,
            'mixing_device: Optional[str]': self.mixing_device if self.mixing_device else None,
            'mixing_media: Optional[str]': self.mixing_media if self.mixing_media else None,
        }


class Operation:
    def __init__(self, type: str, token: str, conditions: OperationConditions):
        self.type = type
        self.token = token
        self.conditions = conditions

    def __repr__(self):
        return (f"Operation(type={self.type!r}, token={self.token!r}, "
                f"conditions={self.conditions!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def default(self, o):
        return o.__dict__
    
    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'type: str': self.type,
            'token: str': self.token,
            'conditions: OperationConditions': self.conditions,
        }


class ReactionEntry:
    def __init__(self, doi: str, paragraph_string: str, synthesis_type: str, reaction_string: str, reaction: Formula, targets_string: List[str], target: Material, precursors: List[Material], operations: List[Operation]):
        self.doi = doi
        self.paragraph_string = paragraph_string
        self.synthesis_type = synthesis_type
        self.reaction_string = reaction_string
        self.reaction = reaction
        self.targets_string = targets_string
        self.target = target
        self.precursors = precursors
        self.operations = operations

    def __repr__(self):
        return (f"ReactionEntry(doi={self.doi!r}, paragraph_string={self.paragraph_string!r}, "
                f"synthesis_type={self.synthesis_type!r}, reaction_string={self.reaction_string!r}, "
                f"reaction={self.reaction!r}, targets_string={self.targets_string!r}, "
                f"target={self.target!r}, precursors={self.precursors!r}, "
                f"operations={self.operations!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)

    def default(self, o):
        return o.__dict__

    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'doi: str': self.doi,
            'paragraph_string: str': self.paragraph_string,
            'synthesis_type: str': self.synthesis_type,
            'reaction_string: str': self.reaction_string,
            'reaction: Formula': self.reaction,
            'targets_string: List[str]': self.targets_string,
            'target: Material': self.target,
            'precursors: List[Material]': [x.to_dict() for x in self.precursors],
            'operations: List[Operation]':[x.to_dict() for x in self.operations],
        }

class Payload:
    def __init__(self, release_date: str, reactions: List[ReactionEntry]):
        self.release_date = release_date
        self.reactions = reactions

    def __repr__(self):
        return f"Payload(release_date={self.release_date!r}, reactions={self.reactions!r})"

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def default(self, o):
        return o.__dict__
    
    def __dict__(self): self.to_dict()
    
    def to_dict(self):
        return {
            'release_date: str': self.release_date,
            'reactions: List[ReactionEntry]': self.reactions,
        }

