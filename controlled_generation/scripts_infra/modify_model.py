from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithCrossAttentions,
)


class WrapperModel(PreTrainedModel, GenerationMixin):
    """
    Ideally take an arbitrary LM head base model and put some arbitrary transformation
    to the head state (and then pass to the lm_head)
    Mostly works with the generation pipeline
    Needs to be registered with AutoModelForCausalLM to use with huggingface pipeline
    """

    def __init__(self, base_model: str, intervention, generation: bool = False):
        """
        base_model: a model with a LM head
        intervention: anything that implies a __call__ model that takes a tensor and returns a tensor of the same dimensions
        """
        self.config = AutoConfig.from_pretrained(base_model)
        self.generation = generation
        super().__init__(self.config)
        self.lm_model = AutoModelForCausalLM.from_pretrained(base_model)
        self.lm_model.eval()

        self.transformer = self.lm_model.transformer
        self.transformer.eval()

        self.lm_head = self.lm_model.lm_head
        self.lm_head.eval()

        self.intervention = intervention

    def can_generate(self):
        return self.generation

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs
    ):
        return self.lm_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

    def forward(self, input_ids, **kwargs):
        # print(self.device)
        base_model_output = self.transformer(input_ids, **kwargs)
        last_hidden_state = base_model_output.last_hidden_state
        new_last_hidden_state = self.intervention(last_hidden_state)

        if self.generation:
            logits = self.lm_head(new_last_hidden_state)

            return CausalLMOutputWithCrossAttentions(
                logits=logits,
                hidden_states=None,
                attentions=base_model_output.attentions,
            )
        else:
            return BaseModelOutputWithCrossAttentions(
                last_hidden_state=new_last_hidden_state,
                hidden_states=None,
                attentions=base_model_output.attentions,
            )
