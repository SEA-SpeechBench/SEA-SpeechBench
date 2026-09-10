def postprocess_text(self, text_list):
    if self.task_category in ["ASR", "ST", "TCQ"]:
        normalization_language = self.target_language if self.task_category in ["ST"] else self.language

        # normalize references and predictions according to normalization_language
        # For now, return text_list as-is for ST tasks
        # TODO: Implement proper text normalization for different languages
        return text_list
    else:
        return text_list