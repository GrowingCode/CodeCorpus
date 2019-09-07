package com.zzk;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Image;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;
import com.itextpdf.text.pdf.PdfWriter;

public class AddWatermark {
	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream(
					"c:\\tempWatermark.pdf"));// 关联文档对象与临时文件的输出流
			document.open();// 打开文档
			document.add(new Paragraph(" ")); // 向文档中添加内容
			document.close();// 关闭文档对象
			PdfReader reader = new PdfReader("c:\\tempWatermark.pdf");// 创建“tempWatermark.pdf”的PdfReader对象
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
					"c:\\添加水印.pdf"));// 创建PdfStamper对象
			Image img = Image.getInstance("image/watermark.jpg");// 创建图像对象
			img.setAbsolutePosition(50, 385);// 定位图片对象
			PdfContentByte under = stamp.getUnderContent(1);// 获得第一页的内容
			under.addImage(img);// 添加图片,完成水印功能
			BaseFont chinese = BaseFont.createFont("STSong-Light",
					"UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义字体
			under.beginText();// 标记文本开始
			under.setFontAndSize(chinese, 42);// 设置字体和字号
			under.setTextMatrix(70, 550);// 设置添加内容的显示位置
			under.showText("下面是添加的水印图片.");// 添加内容
			under.endText();// 标记文本结束
			stamp.close();// PdfStamper对象，将从“tempWatermark.pdf”中读取的文档添加水印后写入“添加水印.pdf”
			File file = new File("c:\\tempWatermark.pdf");// 创建临时文件的File对象
			file.delete();// 删除临时文件
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
