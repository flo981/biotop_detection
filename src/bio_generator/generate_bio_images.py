import geopandas as gpd
import folium
from folium.folium import Map
from pyproj import Transformer
from selenium import webdriver
import os
import io
from PIL import Image
from folium.features import DivIcon


def getText(temp_biotop):
    temp_nummer = temp_biotop['Nummer'][temp_biotop['Nummer'].keys()[0]]
    temp_gemnr = temp_biotop['gemnr'][temp_biotop['gemnr'].keys()[0]]
    temp_biotoptyp = temp_biotop['biotoptyp'][temp_biotop['biotoptyp'].keys()[0]]
    temp_bezeich = temp_biotop['bezeich'][temp_biotop['bezeich'].keys()[0]]
    text = "Nummer: " + str(temp_nummer) + "\n" + "GemNr: " + str(temp_gemnr) + "\n" + "Biotoptyp: " + str(
        temp_biotoptyp) + "\n" + "Bezeichnung: " + str(temp_bezeich)
    return text

def style_fcn(x):
    return {'stroke':True,'color': '#FF0000', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}

def style_fnc_mask(x):
    return {'stroke':True,'color': '#FFFFFF', 'fillColor': '#FFFFFF','opacity':1, 'weight':10, 'line_cap':'round', 'fill':True, 'fillOpacity': 1}

def style_fcn_hull(x):
    return {'stroke':True,'color': '#008FFF', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}


def style_fcn_env(x):
    return {'stroke':True,'color': '#F700FF', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}



def biotop_center(temp_biotop):
    #get biotop pos
    center_temp = temp_biotop.geometry.centroid
    key = center_temp.keys()[0]
    return transformer.transform(center_temp[key].x,center_temp[key].y)

def biotop_current_map(temp_location, text, BIOTOP_BORDER, BIOTOP_MASK ,BIOTOP_DESCRIPTION, case):
    #TODO:  Linie weiter ausen umrunden lassen oder zusätzliche box
    #       Max Zoom tetsen (changed 18->20)

    BIOTOP_SCALE = BIOTOP_CONVEX = BIOTOP_ENV = 0

    m_temp = Map(tiles=None, location=[temp_location[1], temp_location[0]],
                 zoom_start=20,
                 prefer_canvas=True,
                 no_touch=True,
                 disable_3d=True,
                 attr="test",
                 control_scale=True,
                 zoom_control=False)

    atlas = folium.raster_layers.WmsTileLayer(url=wmts, layers="bmaporthofoto30cm",
                                              fmt='image/png',
                                              transparent=True,
                                              overlay=True).add_to(m_temp)

    if BIOTOP_BORDER:
        case = 'B_'+str(case)
        folium.GeoJson(temp_biotop, style_function=style_fcn).add_to(m_temp)
    # folium.GeoJson(temp_biotop_scaled,style_function=style_fcn).add_to(m_temp)
    # folium.GeoJson(temp_biotop_convex_hull, style_function=style_fcn_hull).add_to(m_temp)
    # folium.GeoJson(temp_biotop_envelope, style_function=style_fcn_env).add_to(m_temp)

    if BIOTOP_MASK:
        case = 'M_'+str(case)
        folium.GeoJson(temp_biotop, style_function=style_fnc_mask).add_to(m_temp)

    ## WRITE BIOTOP DESCIBTION IN IMAGE
    if BIOTOP_DESCRIPTION:
        folium.map.Marker([temp_location[1] + 0.001, temp_location[0] - 0.001],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0),
                        html='<div style="font-size: 10pt;"><span style="color: #ff0000;">%s</span></div>' % text)).add_to(m_temp)
    if BIOTOP_SCALE:
        temp_biotop_scaled = temp_biotop.scale(xfact=2, yfact=2, zfact=0, origin='center')
    if BIOTOP_CONVEX:
        temp_biotop_convex_hull = temp_biotop.convex_hull
    if BIOTOP_ENV:
        temp_biotop_envelope = temp_biotop.envelope

    save_current_biotop2(m_temp, bio_i, case)
    #return m_temp

def save_current_biotop(m_temp, bio_number):
    png = m_temp._to_png()
    out = open(os.path.join(script_dir, '../../data/output_biotop/'+bio_number+'.png'), 'wb')
    bytes_written = out.write(png)
    out.close()

def create_bio_path(bio_i):
    temp_path = '../../data/output_biotop_dir/' + 'bio_' + bio_i + '/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
        return True
    return False

def save_current_biotop2(m_temp, bio_number,case):
    img_data = m_temp._to_png()
    img = Image.open(io.BytesIO(img_data))
    temp_path = '../../data/output_biotop_dir/' + 'bio_' + bio_number + '/'
    img.save(os.path.join(script_dir, temp_path + case + '_' + bio_number + '.png'))

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    print('Load: Biotopkartierung')
    nReserve = gpd.read_file(os.path.join(script_dir, '../../data/Biotopkartierung/Biotopkartierung.shp'))

    biotop_1 = nReserve.query("biotoptyp=='8.1.1.1'")
    biotop_2 = nReserve.query("biotoptyp=='8.1.1.2'")

    print('Load: bmaporthofoto30cm')
    wmts = "http://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"
    #driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver')

    print('Start: webdriver (Chrome)')
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')

    transformer = Transformer.from_crs(31258, 4326, always_xy=True)  # 1frpm 2to

    len_bio_1 = biotop_1.shape[0]
    len_bio_2 = biotop_2.shape[0]
    len_totoal =  len_bio_1 + len_bio_2

    biotop_key_nummer = biotop_1['Nummer']
    biotop_keys = biotop_key_nummer.keys()
    for i in range(0, len_bio_1-1):
        key_i = biotop_keys[i]
        bio_i = biotop_1['Nummer'][key_i]
        str_nummer = "\'" + bio_i + "\'"
        query_str = "Nummer==" + str(str_nummer)
        temp_biotop = biotop_1.query(query_str)
        temp_location = biotop_center(temp_biotop)

        PATH_NEW = create_bio_path(bio_i)
        if PATH_NEW:
            print("Process Biotop (1): ", i, "/", len_totoal, " ", bio_i)
            #m_temp = biotop_current_map(temp_location,text=getText(temp_biotop),BIOTOP_BORDER=False, BIOTOP_DESCRIPTION=False, case=1)
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=True,
                           case='1')
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=True, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=False,
                           case=1)
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = True, BIOTOP_DESCRIPTION=False,
                           case=1)
        #save_current_biotop2(m_temp, bio_i, 1)
        else:
            print("Already Existing Biotop (1): ", i, "/", len_totoal, " ", bio_i)


    biotop_key_nummer = biotop_2['Nummer']
    biotop_keys = biotop_key_nummer.keys()
    for j in range(0, len_bio_2-1):
        key_i = biotop_keys[j]
        bio_i = biotop_2['Nummer'][key_i]
        str_nummer = "\'" + bio_i + "\'"
        query_str = "Nummer==" + str(str_nummer)
        temp_biotop = biotop_2.query(query_str)

        temp_location = biotop_center(temp_biotop)

        PATH_NEW = create_bio_path(bio_i)
        if PATH_NEW:
            print("Process Biotop (2): ", j + i, "/", len_totoal, " ", bio_i)
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=True,
                           case='2')
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=True, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=False,
                           case=2)
            biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = True, BIOTOP_DESCRIPTION=False,
                               case=2)

        #m_temp = biotop_current_map(temp_location,text=getText(temp_biotop),BIOTOP_BORDER=False, BIOTOP_DESCRIPTION=False)
        #save_current_biotop2(m_temp, bio_i, 2)
        else:
            print("Already Existing Biotop (2): ", i, "/", len_totoal, " ", bio_i)

#TODO: down-size images