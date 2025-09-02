def pearson_correlation (x , y ) :
    """
    Calcula el coeficiente de correlacion de Pearson entre dos arrays

    Parameters :
    x, y: arrays de igual longitud

    Returns :
    r: coeficiente de correlacion de Pearson
    """
    # Verificar que tienen la misma longitud
    if len( x ) != len( y ) :
        raise ValueError ("Los arrays deben tener la misma longitud ")

     n = len( x )

    # Calcular medias
    mean_x = np . mean ( x )
    mean_y = np . mean ( y )

    # Calcular numerador y denominador
    numerator = np .sum (( x - mean_x ) * ( y - mean_y ) )
    denominator = np . sqrt ( np .sum (( x - mean_x ) **2) * np .sum (( y - mean_y ) **2) )

    # Evitar division por cero
    if denominator == 0:
        return 0
    return numerator / denominator