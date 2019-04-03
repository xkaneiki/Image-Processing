DELAY:
    MOV R1,#200
    D1:
        MOV R3,#50
    D2:
        MOV R3,#48
    DJNZ R3,$
    DJNZ R2,D2
    DJNZ R1,D1