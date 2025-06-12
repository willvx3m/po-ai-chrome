const result = [];
for(var A = 1; A <= 3; A ++){
    for (var B = A + 1; B <= 10; B ++){
        for( var C = B + 1; C <= 20; C ++){
            const R = 35.2 * A + 20.8 * B - 20.1 * C;
            if(R > 0){
                const F = R / (A+B+C);
                result.push({A, B, C, R, F});
            }
        }
    }
}

result.sort((a, b) => a.F - b.F);
console.log(result.length);
result.forEach(item => {
    console.log(item.A, item.B, item.C, ' :: ', item.F, ' - ', item.R);
});