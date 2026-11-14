# HC up register lookahead

HC up always loads five ascending K16 operand sets per canonical quarter before
applying their unchanged MMA chains. One warp owns each N8 tile. The old kernel
and experiment selector are removed. GPU receipt 895443f reports 5.11 to 3.99
ms/token, full rows equal across four cards at C1/C8 and three repeats. Wider
context, composed-kernel and lookup qualification remain integration gates.
