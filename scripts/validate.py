import logging
import re


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None).lower()
    return str(value).lower()


def split_sentences(text):
    text = str(text or "").replace("\n", " ")
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s and s.strip()]


def unique_preserve_order(values):
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def normalize_evidence(value):
    if value is None:
        return []

    items = value if isinstance(value, list) else [value]
    cleaned = []

    for item in items:
        text = str(item).strip().replace("\n", " ")
        if text and text not in cleaned:
            cleaned.append(text)

    return cleaned


def contains_term(text, term):
    text = value_to_text(text)
    term = str(term or "").lower().strip()

    if not term:
        return False

    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None

    return term in text


def sentence_contains(sentence, terms):
    return any(contains_term(sentence, term) for term in terms or [])


def flatten_normalization_map(section):
    flattened = {}

    for canonical, aliases in (section or {}).items():
        flattened[clean_text(canonical)] = canonical
        for alias in aliases or []:
            flattened[clean_text(alias)] = canonical

    return flattened


def normalize_status(value, vocabulary, default_status):
    status_map = flatten_normalization_map(
        vocabulary.get("normalization", {}).get("statuses", {})
    )
    return status_map.get(clean_text(value), default_status)


def parse_number_text(value, vocabulary):
    value = clean_text(value)

    try:
        return float(value)
    except ValueError:
        pass

    number_words = vocabulary.get("number_words", {})
    if value in number_words:
        return float(number_words[value])

    return None


def get_negated_spans(sentence, vocabulary):
    settings = vocabulary.get("negation_scope", {})
    triggers = settings.get("triggers", [])
    boundary_terms = settings.get("boundary_terms", [])
    max_chars = int(settings.get("max_scope_characters", 260))

    sentence_string = str(sentence or "")
    lower_sentence = clean_text(sentence_string)

    trigger_patterns = sorted(
        [
            re.escape(str(trigger).lower())
            for trigger in triggers
            if str(trigger).strip()
        ],
        key=len,
        reverse=True,
    )

    if not trigger_patterns:
        return []

    trigger_regex = r"\b(?:" + "|".join(trigger_patterns) + r")\b"
    spans = []

    for match in re.finditer(trigger_regex, lower_sentence, flags=re.IGNORECASE):
        start = match.start()
        end = min(len(sentence_string), match.end() + max_chars)
        candidate_lower = lower_sentence[match.end():end]

        boundary_positions = []

        for boundary in boundary_terms:
            boundary_match = re.search(
                rf"\b{re.escape(str(boundary).lower())}\b",
                candidate_lower,
            )
            if boundary_match:
                boundary_positions.append(boundary_match.start())

        if boundary_positions:
            end = match.end() + min(boundary_positions)

        spans.append(sentence_string[start:end].strip())

    return spans


def term_is_negated(sentence, term, vocabulary):
    return any(
        contains_term(span, term)
        for span in get_negated_spans(sentence, vocabulary)
    )


def sentence_is_excluded(sentence, rules):
    return sentence_contains(sentence, rules.get("exclude_terms", []))


def sentence_is_invalid(sentence, rules):
    return sentence_contains(sentence, rules.get("invalid_if_terms", []))


def sentence_meets_candidate_rules(sentence, rules):
    must_contain = rules.get("sentence_must_contain", [])
    cues = rules.get("sentence_cues", [])

    if must_contain and not all(
        contains_term(sentence, term) for term in must_contain
    ):
        return False

    if cues and not any(contains_term(sentence, cue) for cue in cues):
        return False

    return True


def candidate_sentences(note_text, rules):
    output = []

    for sentence in split_sentences(note_text):
        if sentence_is_excluded(sentence, rules):
            continue
        if sentence_is_invalid(sentence, rules):
            continue
        if sentence_meets_candidate_rules(sentence, rules):
            output.append(sentence)

    return output


def score_sentence(sentence, preferred_terms):
    return sum(1 for term in preferred_terms or [] if contains_term(sentence, term))


def choose_best_sentence(sentences, preferred_terms=None):
    if not sentences:
        return ""

    if not preferred_terms:
        return sentences[0]

    return sorted(
        sentences,
        key=lambda sentence: score_sentence(sentence, preferred_terms),
        reverse=True,
    )[0]


def apply_template(template, match):
    value = str(template)

    for name, group_value in match.groupdict().items():
        value = value.replace("{" + name + "}", group_value or "")

    return value


def apply_field_assignments(validated, field_values, match=None):
    for field_name, template in (field_values or {}).items():
        if field_name not in validated:
            continue

        if match is not None:
            validated[field_name] = apply_template(template, match)
        else:
            validated[field_name] = template


def find_category_matches(note_text, extractor, vocabulary):
    matches = []
    evidence_by_category = {}

    ignore_negated = extractor.get("ignore_negated", False)
    categories = extractor.get("categories", {})

    for category, rule in categories.items():
        if not isinstance(rule, dict):
            continue

        terms = rule.get("terms", [])
        preferred_terms = rule.get("preferred_evidence_terms", [])
        category_sentences = []

        for sentence in split_sentences(note_text):
            if sentence_is_excluded(sentence, rule):
                continue
            if sentence_is_invalid(sentence, rule):
                continue

            for term in terms:
                if not contains_term(sentence, term):
                    continue

                if ignore_negated and term_is_negated(sentence, term, vocabulary):
                    continue

                category_sentences.append(sentence)
                break

        if category_sentences:
            matches.append(category)
            evidence_by_category[category] = choose_best_sentence(
                category_sentences,
                preferred_terms,
            )

    return unique_preserve_order(matches), evidence_by_category


def apply_categorical_extractor(validated, note_text, extractor, vocabulary, settings):
    output_field = extractor.get("output_field")
    evidence_field = extractor.get("evidence_field")
    default = extractor.get("default", "")
    allow_multiple_as = extractor.get("allow_multiple_as")
    overwrite = extractor.get("overwrite", True)

    if not output_field or output_field not in validated:
        return

    if not overwrite and validated.get(output_field):
        return

    matches, evidence_by_category = find_category_matches(
        note_text,
        extractor,
        vocabulary,
    )

    if not matches:
        if default:
            validated[output_field] = default
        return

    priority = extractor.get("priority", [])
    ordered_matches = [item for item in priority if item in matches] or matches

    if len(ordered_matches) > 1 and allow_multiple_as:
        selected = allow_multiple_as
    else:
        selected = ordered_matches[0]

    validated[output_field] = selected

    if evidence_field and evidence_field in validated:
        evidence = evidence_by_category.get(selected)

        if not evidence and ordered_matches:
            evidence = evidence_by_category.get(ordered_matches[0])

        if evidence:
            validated[evidence_field] = [evidence]


def apply_multi_categorical_extractor(validated, note_text, extractor, vocabulary, settings):
    output_field = extractor.get("output_field")
    evidence_field = extractor.get("evidence_field")
    overwrite = extractor.get("overwrite", True)

    if not output_field or output_field not in validated:
        return

    if not overwrite and validated.get(output_field):
        return

    matches, evidence_by_category = find_category_matches(
        note_text,
        extractor,
        vocabulary,
    )

    priority = extractor.get("priority", [])
    if priority:
        matches = [item for item in priority if item in matches]

    validated[output_field] = matches

    if evidence_field and evidence_field in validated:
        preferred_terms = extractor.get("preferred_evidence_terms", [])
        evidence_sentences = [
            evidence_by_category[item]
            for item in matches
            if item in evidence_by_category
        ]
        evidence = choose_best_sentence(evidence_sentences, preferred_terms)

        if evidence:
            validated[evidence_field] = [evidence]


def get_regex_candidate_pool(validated, note_text, extractor):
    evidence_field = extractor.get("evidence_field")

    candidates = candidate_sentences(note_text, extractor)

    fallback_sentences = []

    if evidence_field:
        fallback_sentences.extend(
            normalize_evidence(validated.get(evidence_field, ""))
        )

    fallback_sentences.extend(split_sentences(note_text))

    return unique_preserve_order(candidates + fallback_sentences)


def apply_regex_fields_extractor(validated, note_text, extractor, vocabulary, settings):
    evidence_field = extractor.get("evidence_field")
    preferred_terms = extractor.get("preferred_evidence_terms", [])

    sentences = get_regex_candidate_pool(
        validated=validated,
        note_text=note_text,
        extractor=extractor,
    )

    for pattern_rule in extractor.get("patterns", []) or []:
        regex = pattern_rule.get("regex")
        field_values = pattern_rule.get("fields", {})

        if not regex:
            continue

        matching_sentences = []

        for sentence in sentences:
            match = re.search(regex, sentence, flags=re.IGNORECASE)
            if match:
                matching_sentences.append((sentence, match))

        if not matching_sentences:
            continue

        best_sentence = choose_best_sentence(
            [item[0] for item in matching_sentences],
            preferred_terms,
        )

        best_match = None

        for sentence, match in matching_sentences:
            if sentence == best_sentence:
                best_match = match
                break

        apply_field_assignments(validated, field_values, best_match)

        if evidence_field and evidence_field in validated and best_sentence:
            validated[evidence_field] = [best_sentence]

        return


def threshold_rule_matches(sentence, rule, vocabulary):
    if not sentence_meets_candidate_rules(sentence, rule):
        return False

    for pattern in rule.get("patterns", []) or []:
        match = re.search(pattern, clean_text(sentence), flags=re.IGNORECASE)

        if not match:
            continue

        number_text = match.groupdict().get("number")

        if number_text is None and match.groups():
            number_text = match.group(1)

        number = parse_number_text(number_text, vocabulary)

        if number is None:
            continue

        comparator = rule.get("comparator")
        threshold = float(rule.get("value"))

        if comparator == ">=" and number >= threshold:
            return True
        if comparator == ">" and number > threshold:
            return True
        if comparator == "<=" and number <= threshold:
            return True
        if comparator == "<" and number < threshold:
            return True
        if comparator == "==" and number == threshold:
            return True

    return False


def apply_threshold_status_extractor(validated, note_text, extractor, vocabulary, settings):
    status_field = extractor.get("status_field")
    evidence_field = extractor.get("evidence_field")

    if not status_field or status_field not in validated:
        return

    for status_name in ["absent", "present"]:
        status_rule = extractor.get(status_name, {})
        sentences = candidate_sentences(note_text, status_rule)

        for sentence in sentences:
            if threshold_rule_matches(sentence, status_rule, vocabulary):
                validated[status_field] = status_name

                if evidence_field and evidence_field in validated:
                    validated[evidence_field] = [sentence]

                return


def apply_status_extractor(validated, note_text, extractor, vocabulary, settings):
    status_field = extractor.get("status_field")
    evidence_field = extractor.get("evidence_field")

    if not status_field or status_field not in validated:
        return

    absent_terms = extractor.get("absent_terms", [])
    present_terms = extractor.get("present_terms", [])

    for sentence in split_sentences(note_text):
        if sentence_is_excluded(sentence, extractor):
            continue
        if sentence_is_invalid(sentence, extractor):
            continue

        if sentence_contains(sentence, absent_terms):
            validated[status_field] = "absent"

            if evidence_field and evidence_field in validated:
                validated[evidence_field] = [sentence]

            return

        for span in get_negated_spans(sentence, vocabulary):
            if sentence_contains(span, present_terms + absent_terms):
                validated[status_field] = "absent"

                if evidence_field and evidence_field in validated:
                    validated[evidence_field] = [sentence]

                return

    for sentence in split_sentences(note_text):
        if sentence_is_excluded(sentence, extractor):
            continue
        if sentence_is_invalid(sentence, extractor):
            continue

        for term in present_terms:
            if contains_term(sentence, term) and not term_is_negated(
                sentence,
                term,
                vocabulary,
            ):
                validated[status_field] = "present"

                if evidence_field and evidence_field in validated:
                    validated[evidence_field] = [sentence]

                return


EXTRACTOR_HANDLERS = {
    "categorical": apply_categorical_extractor,
    "multi_categorical": apply_multi_categorical_extractor,
    "regex_fields": apply_regex_fields_extractor,
    "threshold_status": apply_threshold_status_extractor,
    "status": apply_status_extractor,
}


def apply_extractors(validated, note_text, vocabulary, settings):
    for extractor_name, extractor in (vocabulary.get("extractors") or {}).items():
        extractor_type = extractor.get("type")
        handler = EXTRACTOR_HANDLERS.get(extractor_type)

        if not handler:
            logging.warning(
                "Unknown extractor type '%s' for extractor '%s'.",
                extractor_type,
                extractor_name,
            )
            continue

        handler(validated, note_text, extractor, vocabulary, settings)


def evidence_is_exact_or_empty(evidence, note_text):
    if not evidence:
        return True

    note = str(note_text or "")

    for item in normalize_evidence(evidence):
        if item and item not in note:
            return False

    return True


def repair_non_exact_evidence(validated, note_text):
    if not note_text:
        return

    for column_name in list(validated.keys()):
        if not column_name.endswith("_evidence"):
            continue

        if not evidence_is_exact_or_empty(validated.get(column_name), note_text):
            validated[column_name] = []


def repair_absent_without_evidence(validated, csv_columns):
    for column_name in csv_columns:
        if not column_name.endswith("_status"):
            continue

        evidence_column = column_name.replace("_status", "_evidence")

        if evidence_column not in csv_columns:
            continue

        if validated.get(column_name) == "absent" and not validated.get(evidence_column):
            validated[column_name] = "not_listed"
            validated[evidence_column] = []


def normalize_initial_values(validated, csv_columns, extraction, settings, vocabulary, note_id):
    validation = settings.get("validation") or {}
    default_status = validation.get("default_status", "not_listed")
    allowed_statuses = set(
        validation.get("allowed_statuses", ["present", "absent", "not_listed"])
    )

    for column_name in csv_columns:
        if column_name in extraction:
            validated[column_name] = extraction[column_name]

    validated["note_id"] = note_id

    for column_name in csv_columns:
        if column_name.endswith("_status"):
            status = normalize_status(
                validated.get(column_name),
                vocabulary,
                default_status,
            )
            validated[column_name] = status if status in allowed_statuses else default_status

    for column_name in csv_columns:
        if column_name.endswith("_evidence"):
            validated[column_name] = normalize_evidence(validated.get(column_name))


def blank_not_listed_evidence(validated, csv_columns):
    for column_name in csv_columns:
        if not column_name.endswith("_status"):
            continue

        evidence_column = column_name.replace("_status", "_evidence")

        if evidence_column in csv_columns and validated.get(column_name) == "not_listed":
            validated[evidence_column] = []


def log_blank_fields(validated, note_id):
    blank_fields = []

    for field_name, value in validated.items():
        if field_name == "note_id":
            continue

        if value is None:
            blank_fields.append(field_name)
        elif value == "":
            blank_fields.append(field_name)
        elif isinstance(value, list) and not value:
            blank_fields.append(field_name)

    if blank_fields:
        logging.info(
            "Blank fields after validation for note_id=%s: %s",
            note_id,
            ", ".join(blank_fields),
        )


def validate_extraction(
    extraction,
    csv_columns,
    settings,
    vocabulary,
    note_id,
    note_text="",
):
    if not isinstance(extraction, dict):
        raise ValueError("Extraction output must be a JSON object.")

    validated = dict(csv_columns)

    normalize_initial_values(
        validated=validated,
        csv_columns=csv_columns,
        extraction=extraction,
        settings=settings,
        vocabulary=vocabulary,
        note_id=note_id,
    )

    apply_extractors(validated, note_text, vocabulary, settings)
    repair_non_exact_evidence(validated, note_text)
    repair_absent_without_evidence(validated, csv_columns)
    blank_not_listed_evidence(validated, csv_columns)
    log_blank_fields(validated, note_id)

    return validated