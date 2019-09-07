package com.zzk;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class AddKeywords {
	public static void main(String[] args){
		Document document=new Document();// 创建文档对象
		 try {
			PdfWriter.getInstance(document, new FileOutputStream("C:\\Java编程词典.pdf"));//关联文档对象与输出流
			document.addKeywords("这是一套Java开发人员必备的学习资源库!");// 向文档中添加关键字	
			document.open();// 打开文档
			document.add(new Paragraph("Keywords"));// 向文档中添加内容	
	        document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	   }
}
