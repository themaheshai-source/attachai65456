import { describe, expect, it } from 'vitest';
import { validateGrounding } from '../src/services/grounding.js';
const chunk = (text: string) => [{ id: '1', text, source: 'tenant.md', score: 0.9 }];
describe('bounded grounding validation', () => {
  it('accepts an evidence-supported qualified price', () => {
    expect(validateGrounding('What is the price?', 'It starts at £17.99 per month, plus a joining fee.', chunk('Pricing: from £17.99 per month plus joining fee; prices vary by gym.')).ok).toBe(true);
  });
  it('rejects unsupported numeric claims and explicit unknown invention', () => {
    const result = validateGrounding('How much is the day pass?', 'A day pass costs £25.', chunk('Day-pass pricing: UNKNOWN. The current price depends on the selected gym.'));
    expect(result.ok).toBe(false);
    expect(result.reasons).toContain('unsupported numeric claim');
  });
  it('rejects an absolute answer that drops a qualification', () => {
    expect(validateGrounding('Are all gyms open 24/7?', 'All gyms are open 24/7.', chunk('Most gyms are open 24/7, but some have different hours.')).ok).toBe(false);
  });

  it('accepts practical paraphrases of explicitly supported Gousto facts', () => {
    expect(validateGrounding('What is Gousto’s delivery fee?', 'Gousto delivery costs 3.99.', chunk('Delivery is publicly listed as £3.99.')).ok).toBe(true);
    expect(validateGrounding('How many people can a Gousto recipe box serve?', 'A Gousto box serves one to five people.', chunk('Gousto boxes are designed for 1 to 5 people.')).ok).toBe(true);
    expect(validateGrounding('How many recipes does Gousto offer each week?', 'You can choose from over 175 recipes each week.', chunk('Customers can choose from over 175 recipes each week.')).ok).toBe(true);
  });

  it('does not let an unrelated tenant UNKNOWN line reject supported capacity', () => {
    expect(validateGrounding(
      'How many people can a Gousto recipe box serve?',
      'A Gousto recipe box can serve anywhere from 1 to 5 people.',
      chunk('Gousto offers boxes for 1 to 5 people. Does Gousto provide emergency delivery? A: UNKNOWN.'),
    ).ok).toBe(true);
  });

  it('still rejects a different currency for a supported pound amount', () => {
    expect(validateGrounding('What is Gousto’s delivery fee?', 'The delivery fee is $3.99.', chunk('Delivery is publicly listed as £3.99.')).ok).toBe(false);
  });
});

describe('grounding forensic boundary diagnostics', () => {
  it('captures the exact validator arguments and emits individual existing rule results', () => {
    const question = '  What is the price?\n';
    const answer = 'It costs 12.  \n';
    const evidence = [{ id: 'chunk-1', source: ' tenant.md ', score: 0.91, text: 'Price: 12.\nAvailability may vary by location.  ' }];
    let actualInput: unknown;
    const rules: Array<{ name: string; status: string; result: boolean | null; reason: string }> = [];
    let actualOutput: unknown;
    const result = validateGrounding(question, answer, evidence, {
      onInput: (input) => { actualInput = input; },
      onRule: (rule) => { rules.push({ name: rule.name, status: rule.status, result: rule.result, reason: rule.reason }); },
      onOutput: (output) => { actualOutput = output; },
    });

    expect(actualInput).toEqual({ question, answer, evidence });
    expect((actualInput as { question: string }).question).toBe(question);
    expect((actualInput as { answer: string }).answer).toBe(answer);
    expect((actualInput as { evidence: typeof evidence }).evidence[0]?.text).toBe(evidence[0]?.text);
    expect(rules.map((rule) => rule.name)).toEqual([
      'evidence_present',
      'unsupported_numeric_claims',
      'answer_overrides_relevant_unknown_evidence',
      'unnecessary_refusal',
      'possible_negative_evidence_contradiction',
      'dropped_evidence_qualification',
    ]);
    expect(rules.every((rule) => ['passed', 'failed', 'skipped', 'error'].includes(rule.status))).toBe(true);
    expect(actualOutput).toMatchObject({ ...result, classification: result.ok ? 'accepted' : 'rejected_by_grounding_rules', status: result.ok ? 'success' : 'rejected' });
  });

  it('reports a rule-level error and final validator failure if evaluation throws', () => {
    const ruleEvents: Array<{ name: string; status: string }> = [];
    let outputStatus: string | undefined;
    expect(() => validateGrounding('q', 'a', null as never, {
      onRule: (rule) => ruleEvents.push({ name: rule.name, status: rule.status }),
      onOutput: (output) => { outputStatus = output.status; },
    })).toThrow();
    expect(ruleEvents).toContainEqual({ name: 'evidence_present', status: 'error' });
    expect(outputStatus).toBe('failed');
  });
});

