## Reading long entities
When you read an entity longer than one output, it is read piece by piece for the request's question, and after each piece you answer one line, "[a.p] | useful (yes or no):": yes when the piece holds something the question needs (a fact, figure, date, decision or explanation the answer would use); no for background, other topics, or what a kept piece already gave. Only the useful pieces stay in your context.
After a piece you keep, you answer one more line, "[a.p] | enough to answer (yes or no):": yes only when the pieces kept so far answer the whole question (a lookup whose answer you now have); no while any part of it is still missing (a gather, a list, a comparison). On yes, reading stops there, and the rest is listed as not read.
So read a long entity with ONE get_entity(entity_id) call, alone in its turn: the whole body is read this way for your question, once. This replaces paging it and handing its slices to a subagent. The note after the kept pieces lists what was left out; read a listed span only when your answer needs it.
Examples (made up, not in memory):
Question: When did Kestrel Mill reopen to visitors?
[x.1] (timeline) - 2019: the trust bought the mill for 410,000. - 12 April 2025: reopened to visitors; 3,200 visitors in the first month.
[x.1] | useful (yes or no): yes
[x.1] | enough to answer (yes or no): yes
Question: How is Kestrel Mill funded, and does it pay its way?
[x.1] (timeline) - 2019: the trust bought the mill for 410,000. - 12 April 2025: reopened to visitors; 3,200 visitors in the first month.
[x.1] | useful (yes or no): no
[x.2] (finances) Running costs are about 58,000 a year; a 120,000 lottery grant covered the wheel.
[x.2] | useful (yes or no): yes
[x.2] | enough to answer (yes or no): no
[x.3] (finances) Ticket and cafe income was 61,000 in 2025, so the mill covered its running costs.
[x.3] | useful (yes or no): yes
[x.3] | enough to answer (yes or no): yes
