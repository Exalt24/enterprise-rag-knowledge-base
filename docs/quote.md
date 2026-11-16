# Quotation drafter

`app/quote/` drafts a quotation from an enquiry, a fee schedule with business rules, and a template, and flags every doubt for a person. It is plain arithmetic and plain rules, so a figure on the draft is never a model's guess, and nothing is ever sent. It lives beside the [two-PDF check](compare.md) and shares its idea: the tool does what can be checked, a named reviewer decides the rest, and the approval is bound to a hash.

The inputs in [quote-sample/](quote-sample/) are invented: a small consultancy's fee schedule, an enquiry that asks for a service that is not on it, and a template. `quote-sample/make_quote_samples.py` writes the outputs, and a test checks the committed files are exactly what it writes.

## Run it

```bash
cd backend
python -m app.quote draft --enquiry ../docs/quote-sample/enquiry.txt --fees ../docs/quote-sample/fee_schedule.json \
    --template ../docs/quote-sample/template.txt --client "Marlow Council" --reference Q-1001 --discount 15 --out quote.json
python -m app.quote approve quote.json --reviewer "Dana Planner" --acknowledgements ../docs/quote-sample/acknowledgements.json --out approved.json
python -m app.quote verify approved.json
```

What the sample draft prints (`quote-sample/quote.txt`):

```
SI-01    Site inspection                           2 visit  x     850.00 =      1700.00
CR-02    Condition report                          1 report x    1450.00 =      1450.00
DD-03    Design drawing set                       12 sheet  x     620.00 =      7440.00
-        geotechnical investigation                1        x          - =     NO PRICE
CE-04    Certification letter                      0 letter x     380.00 =     NO PRICE

Subtotal   10590.00
Discount   1059.00
GST        953.10
Total      10484.10

DRAFT for review. Open points:
L3: Design drawing set is marked in the fee schedule as needing a person's review
L4: 'geotechnical investigation' is not in the fee schedule, so it has no price
L5: Certification letter has no quantity in the enquiry line 'certification letter x 0'
DISCOUNT: a discount of 15 percent was asked for and the schedule allows 10, so 10 was applied
```

## The rules

- A line is quantity times rate in `Decimal`, rounded half up to the cent, and the totals are sums of those rounded lines. The discount and the tax are rounded the same way, and the tax is on what is left after the discount.
- A service the enquiry names that is not in the fee schedule gets no price and a flag. Quantity below the service's minimum or above its maximum, a service marked as needing review, a line with no quantity, and a total under the minimum quote are each a flag with the reason.
- A requested discount above the schedule's authority is capped at the authority and flagged, never applied in full. Exactly the authority, the maximum or the minimum is allowed.
- The template is filled by name. A placeholder this tool does not know is an error, not a blank in a client's document.
- The fee schedule is validated first: a duplicate code, a tax rate written as a percent, two services that one phrase would match, a maximum below the minimum.
- The enquiry is read from lines such as `3 x site inspection` or `site inspection x 3`, matched on the schedule's own names and aliases. Free text a model has read goes through `extract_with_model`: each service must be in the schedule, carry a plain number and quote words that are really in the enquiry, so a model cannot add a service the client did not ask for.

## The approval step

`approve` needs a named reviewer and one acknowledgement with a reason (at least ten characters) for every flag. The approved file records both and binds them to the quote's hash. `verify` recomputes the hash of the quote and of the approval and re-runs the check, so a changed line, total, flag, reviewer or acknowledgement is found, and a forged approval with a consistent hash but a missing acknowledgement still fails. The command line does not trust the hash written in a quote file it reads: it recomputes it.

## Tests

```bash
cd backend
python -m pytest tests/test_quote.py                    # 144 tests, no service, no key
python tests/mutation_check_compare.py                  # 91 guards for the quote drafter and the PDF check; edits source, so use a copy
```

80 seeded worlds build random fee schedules, enquiries (unknown services, quantities at and beyond the limits, mixed case, both line shapes) and discounts, and check subtotal, discount, tax, total and the set of flags against an oracle on exact fractions that shares no code with the drafter. A coverage test checks the worlds reach a discount flag, a minimum flag, a line flag and a clean draft.

## Limits

- Prices come from the schedule alone: no travel, no staged fees, no tiered rates. A rule the schedule cannot express is a flag for a person.
- The enquiry reader understands only the two line shapes. Anything else is kept as a line with no quantity for a person to read, and the model extractor was tested with a test double, never a live model.
- The tax is one rate for the whole quote, and money is in one currency.
- The sample schedule and enquiry are invented; no real consultancy's fee structure was used.
