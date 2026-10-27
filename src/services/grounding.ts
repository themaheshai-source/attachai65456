import type { RetrievedChunk } from '../domain/types.js';

export type GroundingCheck = { ok: boolean; reasons: string[] };
export type GroundingRuleStatus = 'passed' | 'failed' | 'skipped' | 'error';
export type GroundingRuleDiagnostic = {
  name: string;
  version: 1;
  status: GroundingRuleStatus;
  result: boolean | null;
  reason: string;
  input: Record<string, unknown>;
};
export type GroundingDiagnostics = {
  onInput?: (input: { question: string; answer: string; evidence: RetrievedChunk[] }) => void;
  onRule?: (rule: GroundingRuleDiagnostic) => void;
  onOutput?: (output: GroundingCheck & { classification: string; status?: 'success' | 'rejected' | 'failed' }) => void;
};

type NumericClaim = { key: string; value: string; currency: string; qualifier: string };

const WORD_NUMBERS: Record<string, string> = {
  one: '1',
  two: '2',
  three: '3',
  four: '4',
  five: '5',
  six: '6',
  seven: '7',
  eight: '8',
  nine: '9',
  ten: '10',
};

/**
 * Normalize numeric claims for practical grounding. Currency is retained so a
 * supported pound amount cannot silently become an unsupported dollar amount;
 * omitted currency and harmless spacing/unit wording remain compatible.
 */
const numberTokens = (text: string, includeWordNumbers = false): NumericClaim[] => {
  const claims = [
    ...text.matchAll(
      /(£|AUD\s*\$|\$|€|₹)?\s*(\d+(?:[.,]\d+)?)(?:\s*%|\s*(?:per|a)\s*(?:month|year|day|week))?/gi,
    ),
  ].map((match) => {
    const raw = match[0].toLowerCase();
    const currency = (match[1] ?? '').replace(/\s+/g, '').toLowerCase();
    const value = match[2]!.replace(/,/g, '');
    const qualifier = /%/.test(raw)
      ? '%'
      : /\b(?:per|a)\s*(month|year|day|week)\b/i.exec(raw)?.[1]?.toLowerCase() ?? '';
    return { key: `${currency}:${value}:${qualifier}`, value, currency, qualifier };
  });
  if (includeWordNumbers) {
    for (const [word, digit] of Object.entries(WORD_NUMBERS)) {
      if (new RegExp(`\\b${word}\\b`, 'i').test(text)) {
        claims.push({ key: `:${digit}:`, value: digit, currency: '', qualifier: '' });
      }
    }
  }
  return claims;
};

const words = (text: string) =>
  new Set(
    (text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter(
      (x) =>
        ![
          'what',
          'does',
          'have',
          'with',
          'from',
          'that',
          'this',
          'about',
          'are',
          'the',
          'how',
          'can',
          'you',
          'your',
          'for',
          'and',
        ].includes(x),
    ),
  );

function overlap(a: Set<string>, b: Set<string>) {
  let n = 0;
  for (const x of a) if (b.has(x)) n++;
  return n;
}

const UNNECESSARY_REFUSAL_RE =
  /\b(i don['’]?t have (?:verified |that )?information|sorry,?\s+i (?:have no|don['’]?t have) (?:that )?information|i do not know|unable to find (?:that|any) information)\b/i;

function diagnosticCall<T>(callback: ((value: T) => void) | undefined, value: T) {
  try {
    callback?.(value);
  } catch {
    // Diagnostic callbacks must not affect grounding behavior.
  }
}

/**
 * Conservative, evidence-based checks for the most dangerous grounding failures.
 * It deliberately avoids trying to judge prose style or paraphrase quality.
 * The optional callback observes the exact values at the validator boundary and
 * each existing rule; it does not participate in the decision.
 */
export function validateGrounding(
  question: string,
  answer: string,
  evidence: RetrievedChunk[],
  diagnostics?: GroundingDiagnostics,
): GroundingCheck {
  diagnosticCall(diagnostics?.onInput, { question, answer, evidence });
  const reportRule = (rule: GroundingRuleDiagnostic) => diagnosticCall(diagnostics?.onRule, rule);
  let activeRuleName = 'evidence_present';
  let activeRuleInput: Record<string, unknown> = {};
  const finish = (result: GroundingCheck, classification: string, status: 'success' | 'rejected' = result.ok ? 'success' : 'rejected') => {
    diagnosticCall(diagnostics?.onOutput, { ...result, classification, status });
    return result;
  };

  try {
    activeRuleInput = { evidenceCount: evidence.length };
    if (!evidence.length) {
      reportRule({
        name: 'evidence_present', version: 1, status: 'skipped', result: null,
        reason: 'no_evidence_short_circuit_preserved', input: { evidenceCount: 0 },
      });
      return finish({ ok: true, reasons: [] }, 'accepted_no_evidence_short_circuit');
    }
    reportRule({ name: 'evidence_present', version: 1, status: 'passed', result: true, reason: 'evidence_available', input: { evidenceCount: evidence.length } });
    const source = evidence.map((x) => x.text).join('\n');
    const reasons: string[] = [];
    activeRuleName = 'unsupported_numeric_claims';
    activeRuleInput = { answer, evidenceText: source };
    const answerNumbers = numberTokens(answer, false);
    const sourceNumbers = numberTokens(source, true);
    const unsupported = answerNumbers.filter(
      (claim) =>
        !sourceNumbers.some(
          (supported) =>
            supported.value === claim.value &&
            (!claim.currency || !supported.currency || supported.currency === claim.currency) &&
            (!claim.qualifier || !supported.qualifier || supported.qualifier === claim.qualifier),
        ),
    );
    if (unsupported.length) reasons.push('unsupported numeric claim');
    reportRule({
      name: 'unsupported_numeric_claims', version: 1, status: unsupported.length ? 'failed' : 'passed',
      result: unsupported.length === 0,
      reason: unsupported.length ? 'one_or_more_answer_numbers_not_supported_by_evidence' : 'all_answer_numeric_claims_are_supported_or_no_numeric_claims_exist',
      input: { answer, answerNumbers, evidenceText: source, evidenceNumbers: sourceNumbers, unsupportedClaims: unsupported },
    });

    const qWords = words(question);
    activeRuleName = 'answer_overrides_relevant_unknown_evidence';
    activeRuleInput = { question, answer, evidenceText: source, questionWords: [...qWords] };
    // A tenant/company name alone is not evidence that an UNKNOWN statement is
    // about the requested fact. Require two topical terms, excluding the
    // repeated brand token, to avoid unrelated UNKNOWN entries poisoning valid
    // answers from the same tenant document.
    const topicalQuestionWords = new Set([...qWords].filter((word) => !/^gousto$/i.test(word)));
    const sourceLines = source
      .split(/(?<=[.!?])\s+|\n/)
      .map((x) => x.trim())
      .filter(Boolean);
    const unknownLines = sourceLines.filter((x) =>
      /\bunknown\b|not available|not specified|cannot be verified/i.test(x),
    );
    const hasRelevantUnknown = unknownLines.some(
      (line) => overlap(topicalQuestionWords, words(line)) >= 2,
    );
    const unknownDisclaimer =
      /\b(unknown|not available|not specified|cannot verify|check the official|support)\b/i.test(answer);
    const unknownRuleRejected = Boolean(
      hasRelevantUnknown && answerNumbers.length && !unknownDisclaimer,
    );
    if (unknownRuleRejected) {
      reasons.push('answer supplies a value where relevant evidence is unknown');
    }
    reportRule({
      name: 'answer_overrides_relevant_unknown_evidence', version: 1,
      status: !hasRelevantUnknown || !answerNumbers.length ? 'skipped' : unknownRuleRejected ? 'failed' : 'passed',
      result: !hasRelevantUnknown || !answerNumbers.length ? null : !unknownRuleRejected,
      reason: !hasRelevantUnknown ? 'no_relevant_unknown_evidence' : !answerNumbers.length ? 'answer_has_no_numeric_claim_to_override_unknown' : unknownRuleRejected ? 'numeric_value_given_without_unknown_qualification' : 'unknown_evidence_is_qualified_or_answer_contains_no_unsupported_value',
      input: { question, answer, topicalQuestionWords: [...topicalQuestionWords], unknownLines, hasRelevantUnknown, answerNumbers, unknownDisclaimer, requiredTopicalOverlap: 2 },
    });

    activeRuleName = 'unnecessary_refusal';
    activeRuleInput = { answer, relevantUnknownPresent: hasRelevantUnknown, refusalPattern: UNNECESSARY_REFUSAL_RE.source };
    const unnecessaryRefusal = !hasRelevantUnknown && UNNECESSARY_REFUSAL_RE.test(answer);
    if (unnecessaryRefusal) reasons.push('unnecessary refusal despite available tenant evidence');
    reportRule({
      name: 'unnecessary_refusal', version: 1,
      status: hasRelevantUnknown ? 'skipped' : unnecessaryRefusal ? 'failed' : 'passed',
      result: hasRelevantUnknown ? null : !unnecessaryRefusal,
      reason: hasRelevantUnknown ? 'relevant_unknown_evidence_present' : unnecessaryRefusal ? 'refusal_language_used_despite_non_unknown_evidence' : 'no_unnecessary_refusal_detected',
      input: { answer, refusalPatternMatched: unnecessaryRefusal, relevantUnknownPresent: hasRelevantUnknown, refusalPattern: UNNECESSARY_REFUSAL_RE.source },
    });

    activeRuleName = 'possible_negative_evidence_contradiction';
    activeRuleInput = { question, answer, evidenceText: source };
    const negativeRe = /\b(no|not|does not|do not|never|cannot|must not|without)\b/i;
    const negativeLines = sourceLines.filter((x) => negativeRe.test(x));
    const positiveLines = sourceLines.filter((x) => !negativeRe.test(x));
    const qualifiedLines = sourceLines.filter((x) =>
      /\b(most|some|vary|varies|depends|subject to|may|can differ)\b/i.test(x),
    );
    const affirmative =
      /\b(is|are|does|do|offers?|provides?|guarantees?|available|included|open|free)\b/i.test(
        answer,
      ) && !/\b(not|no|cannot|unavailable|unknown|should not)\b/i.test(answer);
    const aWords = words(answer);
    const minOverlap = Math.min(2, Math.max(1, topicalQuestionWords.size));
    const contradictionComparisons: Array<Record<string, unknown>> = [];
    const contradictsNegative =
      affirmative &&
      negativeLines.some((line) => {
        const lWords = words(line);
        const qHit = overlap(topicalQuestionWords, lWords) >= minOverlap;
        const aHit = overlap(aWords, lWords) >= 1;
        const supportedByPositive = positiveLines.some(
          (pos) => overlap(aWords, words(pos)) > overlap(aWords, lWords),
        );
        contradictionComparisons.push({ line, questionOverlap: overlap(topicalQuestionWords, lWords), answerOverlap: overlap(aWords, lWords), minOverlap, qHit, aHit, supportedByPositive });
        return qHit && aHit && !supportedByPositive;
      });
    if (contradictsNegative) reasons.push('possible contradiction with negative evidence');
    reportRule({
      name: 'possible_negative_evidence_contradiction', version: 1,
      status: contradictsNegative ? 'failed' : 'passed', result: !contradictsNegative,
      reason: contradictsNegative ? 'affirmative_answer_may_conflict_with_topically_relevant_negative_evidence' : 'no_negative_evidence_contradiction_detected',
      input: { question, answer, affirmative, topicalQuestionWords: [...topicalQuestionWords], answerWords: [...aWords], negativeLines, positiveLines, comparisons: contradictionComparisons },
    });

    activeRuleName = 'dropped_evidence_qualification';
    activeRuleInput = { answer, evidenceText: source };
    const answerHasQualification =
      /\b(most|some|vary|varies|depends|subject to|may|can differ|except|however|but)\b/i.test(
        answer,
      );
    const universalClaimPresent = /\b(all|every|always|never)\b/i.test(answer);
    const qualificationComparisons: Array<Record<string, unknown>> = [];
    const matchingQualifiedEvidence = qualifiedLines.some((line) => {
      const questionOverlap = overlap(qWords, words(line));
      qualificationComparisons.push({ line, questionOverlap, minimumOverlap: 1 });
      return questionOverlap >= 1;
    });
    const dropsQualification = universalClaimPresent && !answerHasQualification && matchingQualifiedEvidence;
    if (dropsQualification) reasons.push('answer drops an evidence qualification');
    reportRule({
      name: 'dropped_evidence_qualification', version: 1,
      status: dropsQualification ? 'failed' : 'passed', result: !dropsQualification,
      reason: dropsQualification ? 'unqualified_universal_claim_conflicts_with_qualified_evidence' : 'no_evidence_qualification_drop_detected',
      input: { answer, universalClaimPresent, answerHasQualification, qualifiedLines, comparisons: qualificationComparisons, matchingQualifiedEvidence },
    });

    return finish({ ok: reasons.length === 0, reasons }, reasons.length === 0 ? 'accepted' : 'rejected_by_grounding_rules');
  } catch (error) {
    diagnosticCall(diagnostics?.onRule, {
      name: activeRuleName,
      version: 1,
      status: 'error',
      result: null,
      reason: 'validator_exception_while_evaluating_rule',
      input: activeRuleInput,
    });
    diagnosticCall(diagnostics?.onOutput, {
      ok: false,
      reasons: ['validator_exception'],
      classification: 'validator_exception',
      status: 'failed',
    });
    throw error;
  }
}

export function groundingCorrection(reasons: string[]) {
  return `Grounding check failed (${reasons.join('; ')}). Re-read TENANT KNOWLEDGE as authoritative. Correct or remove every unsupported claim, number, price, availability statement, guarantee, or policy. If TENANT KNOWLEDGE contains the answer, answer directly from it instead of refusing. If the requested fact is absent or marked UNKNOWN, say it is not available in the current tenant knowledge. Return only the corrected customer-facing answer.`;
}
