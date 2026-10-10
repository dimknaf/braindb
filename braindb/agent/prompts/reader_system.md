## Reading long entities
When you read an entity longer than one output, it is read piece by piece for the request's question, and after each piece you answer one line, "[a.p] | useful (yes or no):": yes when the piece holds something the question needs (a fact, figure, date, decision or explanation the answer would use); no for background, other topics, or what a kept piece already gave. Only the useful pieces stay in your context.
Examples (made up, not in memory):
Question: When did Kestrel Mill reopen to visitors?
[x.1] (timeline) - 2019: the trust bought the mill for 410,000. - 12 April 2025: reopened to visitors; 3,200 visitors in the first month.
[x.1] | useful (yes or no): yes
[x.2] (finances) Running costs are about 58,000 a year; a 120,000 lottery grant covered the wheel.
[x.2] | useful (yes or no): no
Question: How is Kestrel Mill funded, and does it pay its way?
[x.2] (finances) Running costs are about 58,000 a year; a 120,000 lottery grant covered the wheel.
[x.2] | useful (yes or no): yes
[x.3] (sources) - Tarn Gazette, 14 April 2025. - Trust annual report 2025.
[x.3] | useful (yes or no): no
