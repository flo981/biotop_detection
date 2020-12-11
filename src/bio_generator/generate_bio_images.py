import argparse
import os
import io
from PIL import Image

import geopandas as gpd

import folium
from folium.folium import Map
from folium.features import DivIcon

from pyproj import Transformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


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

def create_info_textfile(biotop):

    temp_nummer = temp_biotop['Nummer'][temp_biotop['Nummer'].keys()[0]]
    temp_gemnr = temp_biotop['gemnr'][temp_biotop['gemnr'].keys()[0]]
    temp_biotoptyp = temp_biotop['biotoptyp'][temp_biotop['biotoptyp'].keys()[0]]
    temp_bezeich = temp_biotop['bezeich'][temp_biotop['bezeich'].keys()[0]]
    temp_bio_katgem = temp_biotop['bio_katgem'][temp_biotop['bio_katgem'].keys()[0]]
    temp_bio_subnr = temp_biotop['bio_subnr'][temp_biotop['bio_subnr'].keys()[0]]
    temp_bezeichnun = temp_biotop['bezeichnun'][temp_biotop['bezeichnun'].keys()[0]]
    temp_rechtl_Sch = temp_biotop['rechtl_Sch'][temp_biotop['rechtl_Sch'].keys()[0]]
    link1 = temp_biotop['Link'][temp_biotop['Link'].keys()[0]]
    link2 = temp_biotop['Link_ext'][temp_biotop['Link_ext'].keys()[0]]
    #link1: http://anwendung/ins/biotop/display.do?id=1175590
    #link2: https://service.salzburg.gv.at/ins/biotop/disp...
    # =>    https://service.salzburg.gv.at/ins/biotop/ + display.do?id=1175590
    # link immer 7-Stellig? Sonst nach = suchen
    ID = link1[-7:]
    link_public = "https://service.salzburg.gv.at/ins/biotop/display.do?id="+str(ID)
    link_intern = "https://portal.salzburg.gv.at/ins/biotop/display.do?id="+str(ID)

    temp_path = args.output_dir + 'bio_' + temp_nummer + '/' + temp_nummer + "_info.txt"

    output_file = open(temp_path, 'w')
    output_file.write("Biotop Nummer: "+ str(temp_nummer) + "\n")
    output_file.write("Biotop Kat Gem: "+ str(temp_bio_katgem) + "\n")
    output_file.write("Biotop Sub Nr: "+ str(temp_bio_subnr) + "\n")
    output_file.write("Gemeinde Nummer: "+ str(temp_gemnr)+ "\n")
    output_file.write("Biotop Typ: "+ str(temp_biotoptyp) + "\n")
    output_file.write("Biotop Bezeichnung: "+ str(temp_bezeich) + "\n")
    output_file.write("Biotop Bezeichnung(2): "+ str(temp_bezeichnun) + "\n")
    output_file.write("Biotop §: "+ str(temp_rechtl_Sch) + "\n")
    output_file.write("Biotop Link-Public: "+ link_public + "\n")
    output_file.write("Biotop Link-Public: "+ link_intern + "\n")
    output_file.write("Biotop Location: "+ str(biotop_center(biotop)) + "\n")
    output_file.write("Biotop Surface (approx): '{:f}'\n".format(biotop_surface(biotop)))
    output_file.close()
def biotop_surface(biotop):
    #check?
    temp = temp_biotop['geometry'].to_crs({'init': 'epsg:3395'})\
               .map(lambda p: p.area / 10**6)
    return temp[temp.keys()[0]]


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
    temp_path = args.output_dir + 'bio_' + bio_number + '/'
    img.save(os.path.join(script_dir, temp_path + case + '_' + bio_number + '.png'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--shapefile',
        default='../../data/Biotopkartierung/Biotopkartierung.shp',
        help='Path to shapefile')
    parser.add_argument(
        '-b',
        '--biotypes',
        default=['8.1.1.1', '8.1.1.2'],
        nargs="+",
        help='biotypes (list) input must look like : '+str('8.1.1.1' '8.1.1.2' '8.1.2.1'))
    parser.add_argument(
        '-w',
        '--wmts',
        default='http://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg',
        help='wmts layer')
    parser.add_argument(
        '-d',
        '--driver',
        default='chrome',
        help='webdriver (chrome,firefix,safarie)')
    parser.add_argument(
        '-dp',
        '--driver_path',
        default='/usr/local/bin/chromedriver',
        help='path to webdriver')
    parser.add_argument(
        '-o',
        '--output_dir',
        default='../../data/output_biotop_dir/',
        help='out to safe biotop images to')

    parser.add_argument(
        '-m',
        '--mask',
        default=True,
        help='draw mask around biotop')
    parser.add_argument(
        '-t',
        '--text',
        default=True,
        help='draw description into biotop')
    parser.add_argument(
        '-sum',
        '--summary',
        default=True,
        help='make summary')
    args = parser.parse_args()


    script_dir = os.path.dirname(__file__)
    print('Load: Biotopkartierung')
    nReserve = gpd.read_file(os.path.join(script_dir, args.shapefile))

    biotop_dict = {}
    for i in range(0,len(args.biotypes)):
        str_nummer = "\'" + args.biotypes[i] + "\'"
        query_str = "biotoptyp==" + str(str_nummer)
        biotop_dict[i] = nReserve.query(query_str)

    print('Load: bmaporthofoto30cm')
    wmts = args.wmts
    #driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver')

    if args.driver == 'chrome':
        print('Start: webdriver ', args.driver)
        #driver = webdriver.Chrome(executable_path=args.driver_path)
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(args.driver_path, chrome_options=options)
    else:
        print('Different webdriver... break')


    transformer = Transformer.from_crs(31258, 4326, always_xy=True)  # 1frpm 2to


    len_totoal = old_i =  0
    for iter_bio in range(0,len(args.biotypes)):
        len_totoal = len_totoal + biotop_dict[iter_bio].shape[0]

    #for biotop_dict[iter_bio] in biotop_dict:
    for iter_bio in range(0,len(args.biotypes)):
        biotop_key_nummer = biotop_dict[iter_bio]['Nummer']
        biotop_keys = biotop_key_nummer.keys()
        for i in range(0, biotop_dict[iter_bio].shape[0]-1):
            key_i = biotop_keys[i]
            bio_i = biotop_dict[iter_bio]['Nummer'][key_i]
            str_nummer = "\'" + bio_i + "\'"
            query_str = "Nummer==" + str(str_nummer)
            temp_biotop = biotop_dict[iter_bio].query(query_str)
            temp_location = biotop_center(temp_biotop)

            PATH_NEW = create_bio_path(bio_i)
            if PATH_NEW:
                print("Process Biotop: ", i+old_i, "/", len_totoal, " ", bio_i, " type: ", args.biotypes[iter_bio])
                #m_temp = biotop_current_map(temp_location,text=getText(temp_biotop),BIOTOP_BORDER=False, BIOTOP_DESCRIPTION=False, case=1)
                biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=True,
                               case=args.biotypes[iter_bio][-1])
                # biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=True, BIOTOP_MASK = False, BIOTOP_DESCRIPTION=False,
                #                case=args.biotypes[iter_bio][-1])
                # biotop_current_map(temp_location, text=getText(temp_biotop), BIOTOP_BORDER=False, BIOTOP_MASK = True, BIOTOP_DESCRIPTION=False,
                #                case=args.biotypes[iter_bio][-1])

            #save_current_biotop2(m_temp, bio_i, 1)
            #case=biotop_key_nummer[-1] error key!
                if args.text: create_info_textfile(temp_biotop)

            else:
                print("Already Existing Biotop: ", i+old_i, "/", len_totoal, " ", bio_i)
        old_i = old_i + i
    print("Done.")
    driver.quit()



#TODO: down-size images
